from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from tfexample.task import load_model
from tfexample.RLdq import DQN

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

    # --抓取權重--
        # 初始化容器
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
                    float(metrics.get("loss", 0.0)),
                ]
            )
            num_examples.append(int(metrics.get("num-examples", 1)))

        # client 沒有回傳的備案
        if len(weights_list) == 0:  # 若無有效權重，
            return super().aggregate_train(server_round, replies) # 退回原本 FedAvg

    # --建立 KMeans 輸入矩陣(client 特徵組成)--
        X = np.array(feature_list, dtype=np.float64)
        n_clusters = min(3, len(X))
        X_scaled = StandardScaler().fit_transform(X)

    # --執行 KMeans 分群--
        if n_clusters == 1:
            labels = np.zeros(len(X), dtype=int)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)


        print(f"\n Round {server_round} Clustering Result")
        for i, (feat, label, n) in enumerate(zip(feature_list, labels, num_examples)):
          print(
            f"Client {i+1}: "
            f"Cluster={label+1}, "
            f"cos_sim={feat[0]:.4f}, "
            f"l2_norm={feat[1]:.4f}, "
            f"loss={feat[2]:.4f}, "
            f"num_examples={n}"
            )
            
    # --強化學習階段--
        # 計算這一輪的平均特徵，作為當前狀態
        avg_cos = np.mean(X[:, 0])
        avg_l2 = np.mean(X[:, 1])
        avg_loss = np.mean(X[:, 2])
        current_state = np.array([avg_cos, avg_l2, avg_reward])

        # 如果這不是第一輪，代表現在的 avg_reward 是「上一輪 Action」的結果
        if server_round > 1 and self.last_state is not None:
            # 把 (上輪狀態, 上輪動作, 這輪的Reward, 這輪狀態) 存入記憶體並學習
            self.dqn_agent.store_transition(self.last_state, self.last_action, avg_reward, current_state)
            self.dqn_agent.learn()
            
            # 記錄起來供結算畫圖
            self.history_rewards.append(avg_reward)

        # 讓 DQN 根據目前狀態決定這一輪的 Action
        action = self.dqn_agent.choose_action(current_state)
        self.history_actions.append(action)
        print(f"\n[RL Agent] Round {server_round} 選擇策略: Action {action}")

        # 記憶這一輪的狀態與動作，給下一輪算 Reward 用
        self.last_state = current_state
        self.last_action = action

    # --逐群作群內平均--
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

    # --群之間做聚合--
        cluster_weights_ratios = []

        for cluster_id in np.unique(labels):
            idx = np.where(labels == cluster_id)[0]
            
            if action == 0:
                # Action 0: 樣本數加權
                weight = sum([num_examples[i] for i in idx])
            elif action == 1:
                # Action 1: Reward (Loss下降量) 加權
                weight = max(np.mean([X[i, 2] for i in idx]), 0) + 1e-5 
            elif action == 2:
                # Action 2: Cosine Similarity 加權
                weight = max(np.mean([X[i, 0] for i in idx]), 0) + 1e-5

            cluster_weights_ratios.append(weight)

        # 權重歸一化 (變成百分比)
        total_weight = sum(cluster_weights_ratios)
        normalized_weights = [w / total_weight for w in cluster_weights_ratios]
        print(f"  --> 各群體最終分配權重: {[round(w, 4) for w in normalized_weights]}\n")

        # 套用 DQN 決定的權重進行最終 Global Model 聚合
        final_weights = []
        for layer_idx in range(len(cluster_models[0])):
            layer_sum = sum(cluster_models[c_idx][layer_idx] * normalized_weights[c_idx] for c_idx in range(len(cluster_models)))
            final_weights.append(layer_sum)

        # 回傳最後的 global weights
        return ArrayRecord(final_weights), {}


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
