from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from tfexample.task import load_model
from tfexample.RLdq import DQN

# Robust Z Score 標準化
def robust_z_score(values):
    values = np.array(values, dtype=np.float64)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    epsilon = 1e-8
    return 0.6745 * (values - median) / (mad + epsilon)


def positive_part(values):
    return np.maximum(values, 0.0)

app = ServerApp()

# 將 metrics 格式轉換成 dict
def _metric_record_to_dict(metric_record: Any) -> dict:
    if metric_record is None:
        return {}

    if hasattr(metric_record, "to_dict"):
        try:
            return dict(metric_record.to_dict())
        except Exception:
            pass

    if isinstance(metric_record, dict):
        return metric_record

    if hasattr(metric_record, "items"):
        try:
            return dict(metric_record.items())
        except Exception:
            pass

    return {}

# 自訂決策(繼承 FedAvg)
class ClusterStrategy(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 初始化 DQN Agent (3個狀態，3個動作)
        self.dqn_agent = DQN(n_states=3, n_actions=3, n_hidden=50)

        # RL 中 replaybuffer 值的預設
        self.last_state = None
        self.last_action = None

        # 畫圖用的紀錄陣列
        self.history_rewards = []
        self.history_actions = []
    
    def aggregate_train(self, server_round, replies):

    # 抓取權重
        # 初始化
        weights_list = []   # client 回傳的模型權重
        feature_list = []   # client 的特徵
        num_examples = []   # client 的資料量

        # 拆 client 回傳的封包
        for reply in replies:
            arrays = reply.content["arrays"]    # arrays -> 模型權重
            metrics_record = reply.content.get("metrics", None)   # metrics_record -> client 端的指標
            metrics = _metric_record_to_dict(metrics_record)

            # 收集各 client 權重與三特徵
            weights = arrays.to_numpy_ndarrays()
            weights_list.append(weights)

            feature_list.append(
                [
                    float(metrics.get("cos_sim", 0.0)),
                    float(metrics.get("l2_norm", 0.0)),
                    float(metrics.get("delta_loss", 0.0)),
                ]
            )
            num_examples.append(int(metrics.get("num-examples", 1)))

        # client 沒有回傳的備案
        if len(weights_list) == 0:  # 若無有效權重，
            return super().aggregate_train(server_round, replies) # 退回原本 FedAvg

        # 建立 KMeans 輸入矩陣(cluster 特徵組成)
        X = np.array(feature_list, dtype=np.float64)

        cos_vals = X[:, 0]
        l2_vals = X[:, 1]
        delta_loss_vals = X[:, 2]

        # cos_sim 越低越可疑，所以轉成 1 - cos_sim
        direction_deviation = 1.0 - cos_vals

        rz_l2 = positive_part(robust_z_score(l2_vals))
        rz_dir = positive_part(robust_z_score(direction_deviation))
        rz_loss = positive_part(robust_z_score(delta_loss_vals))

        # 異常分數
        anomaly_score = (
            0.4 * rz_l2
            + 0.4 * rz_dir
            + 0.2 * rz_loss
        )

        # KMeans 的輸入用標準化後的三個異常特徵
        X_scaled = np.column_stack([rz_l2, rz_dir, rz_loss])

        # 分成幾群
        n_clusters = min(3, len(X_scaled))

        # --執行 KMeans 分群--
        if n_clusters == 1:
            labels = np.zeros(len(X), dtype=int)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)


        client_levels = []
        for s in anomaly_score:
            if s < 1.5:
                client_levels.append("normal")
            elif s < 3.0:
                client_levels.append("slightly_abnormal")
            else:
                client_levels.append("highly_abnormal")

        print("\n" + "=" * 80)
        print(f"[Round {server_round}] Client Feature / Clustering Summary")
        print("-" * 80)

        for i, (feat, label, n) in enumerate(zip(feature_list, labels, num_examples)):
            print(
                f"Client {i}: "
                f"cluster={label + 1}, "
                f"cos={feat[0]:.4f}, "
                f"l2={feat[1]:.4f}, "
                f"delta_loss={feat[2]:.4f}, "
                f"anomaly={anomaly_score[i]:.4f}, "
                f"level={client_levels[i]}, "
                f"samples={n}"                
            )

         print("=" * 80)
            
    # 強化學習
        # 計算這一輪的平均特徵，作為當前狀態
        avg_rz_l2 = np.mean(rz_l2)
        avg_rz_dir = np.mean(rz_dir)
        avg_rz_loss = np.mean(rz_loss)

        # State 為平均異常特徵
        current_state = np.array([avg_rz_l2, avg_rz_dir, avg_rz_loss], dtype=np.float64)

        # 用平均異常分數作為 reward
        avg_anomaly_score = float(np.mean(anomaly_score))
        rl_reward = -avg_anomaly_score
        
        # 如果這不是第一輪，代表現在的 avg_reward 是「上一輪 Action」的結果
        if server_round > 1 and self.last_state is not None:
            # 把 (上輪狀態, 上輪動作, 這輪的Reward, 這輪狀態) 存入記憶體並學習
            self.dqn_agent.store_transition(
                self.last_state,
                self.last_action,
                rl_reward,
                current_state
            )
            self.dqn_agent.learn()
            self.history_rewards.append(rl_reward)

        # 讓 DQN 根據目前狀態決定這一輪的 Action
        action = self.dqn_agent.choose_action(current_state)
        self.history_actions.append(action)
        action_name = {
            0: "Sample-size weighted aggregation",
            1: "Anomaly-score weighted aggregation",
            2: "Conservative aggregation"
        }

        print(
            f"\n[Round {server_round}] Aggregation Strategy: "
            f"Action {action} - {action_name.get(action, 'Unknown')}"
        )

        # 記憶這一輪的狀態與動作，給下一輪算 Reward 用
        self.last_state = current_state
        self.last_action = action

        # 逐群作群內平均
        cluster_models = []
        for cluster_id in np.unique(labels):
            idx = np.where(labels == cluster_id)[0]
            if len(idx) == 0:
                continue

            cluster_weights = [weights_list[i] for i in idx]  # 抽出此群的模型
            cluster_examples = [num_examples[i] for i in idx]  # 抽出此群的樣本數

            # 算群內總樣本數
            total_examples = sum(cluster_examples)
            if total_examples == 0:
                total_examples = len(cluster_examples)

            # 群內逐層加權平均
            aggregated_cluster = []

            for layer_idx in range(len(cluster_weights[0])):
                layer_sum = sum(
                    cluster_weights[i][layer_idx] * cluster_examples[i]
                    for i in range(len(cluster_weights))
                )
                aggregated_cluster.append(layer_sum / total_examples)

            cluster_models.append(aggregated_cluster)


        if len(cluster_models) == 0:  # 若無得到群模型，
            return super().aggregate_train(server_round, replies) # 退回 FedAvg

        # 群之間做聚合
        cluster_weights_ratios = []

        for cluster_id in np.unique(labels):
            idx = np.where(labels == cluster_id)[0]
            
            if action == 0:
            # Action 0：樣本數加權，偏向資料量大的群
                weight = sum([num_examples[i] for i in idx])

            elif action == 1:
            # Action 1：異常分數加權，異常越高權重越低
                cluster_anomaly = np.mean([anomaly_score[i] for i in idx])
                weight = np.exp(-cluster_anomaly) + 1e-5

            elif action == 2:
            # Action 2：保守策略，只降低高度異常群，其餘保留
                cluster_anomaly = np.mean([anomaly_score[i] for i in idx])
                if cluster_anomaly >= 3.0:
                    weight = 0.1
                elif cluster_anomaly >= 1.5:
                    weight = 0.5
                else:
                    weight = 1.0

            cluster_weights_ratios.append(weight)

        # 權重歸一化 (變成百分比)
        total_weight = sum(cluster_weights_ratios)
        normalized_weights = [w / total_weight for w in cluster_weights_ratios]
        print(
            f"[Round {server_round}] Cluster aggregation weights: "
            f"{[round(w, 4) for w in normalized_weights]}"
        )

        # 套用 DQN 決定的權重進行最終 Global Model 聚合
        final_weights = []
        for layer_idx in range(len(cluster_models[0])):
            layer_sum = sum(cluster_models[c_idx][layer_idx] * normalized_weights[c_idx] for c_idx in range(len(cluster_models)))
            final_weights.append(layer_sum)

        # 回傳最後的 global weights
        return ArrayRecord(final_weights), {}

def aggregate_evaluate(self, server_round, replies):
    print("\n" + "=" * 80)
    print(f"[Round {server_round}] Evaluation Summary")
    print("-" * 80)

    total_correct = 0
    total_examples = 0
    weighted_loss_sum = 0.0

    for i, reply in enumerate(replies):
        metrics_record = reply.content.get("metrics", None)
        metrics = _metric_record_to_dict(metrics_record)

        client_id = int(metrics.get("client_id", i))
        correct_count = int(metrics.get("correct_count", 0))
        total_count = int(metrics.get("total_count", metrics.get("num-examples", 0)))
        eval_acc = float(metrics.get("eval_acc", 0.0))
        eval_loss = float(metrics.get("eval_loss", 0.0))

        total_correct += correct_count
        total_examples += total_count
        weighted_loss_sum += eval_loss * total_count

        print(
            f"Client {client_id}: "
            f"test={total_count}, "
            f"correct={correct_count}, "
            f"acc={eval_acc:.4f}, "
            f"loss={eval_loss:.4f}"
        )

    if total_examples > 0:
        global_acc = total_correct / total_examples
        global_loss = weighted_loss_sum / total_examples

        print("-" * 80)
        print(
            f"Global Evaluation: "
            f"test={total_examples}, "
            f"correct={total_correct}, "
            f"acc={global_acc:.4f}, "
            f"loss={global_loss:.4f}"
        )

    print("=" * 80)

    return super().aggregate_evaluate(server_round, replies)

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]


    model = load_model()
    initial_arrays = ArrayRecord(model.get_weights())


    strategy = ClusterStrategy(
        fraction_train=fraction_train,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
    )


    final_weights = result.arrays.to_numpy_ndarrays()
    model.set_weights(final_weights)
    model.save("final_model.keras")
