from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from tfexample.task import load_model

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

    # KMeans 分群 + 群內聚合 + 群間平均
    def aggregate_train(self, server_round, replies):

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
                    float(metrics.get("reward", 0.0)),
                ]
            )
            num_examples.append(int(metrics.get("num-examples", 1)))

        # client 沒有回傳的備案
        if len(weights_list) == 0:  # 若無有效權重，
            return super().aggregate_train(server_round, replies) # 退回原本 FedAvg

        # 建立 KMeans 輸入矩陣(client 特徵組成)
        X = np.array(feature_list, dtype=np.float64)
        n_clusters = min(5, len(X))

        # 執行 KMeans 分群
        if n_clusters == 1:
            labels = np.zeros(len(X), dtype=int)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)


        print(f"\n Round {server_round} Clustering Result")
        for i, (feat, label, n) in enumerate(zip(feature_list, labels, num_examples)):
          print(
            f"Client {i+1}: "
            f"Cluster={label+1}, "
            f"cos_sim={feat[0]:.4f}, "
            f"l2_norm={feat[1]:.4f}, "
            f"reward={feat[2]:.4f}, "
            f"num_examples={n}"
            )


        cluster_models = []

        # 逐群作群內聚合
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

        # 群間做平均
        final_weights = []

        for layer_idx in range(len(cluster_models[0])):
            layer_sum = sum(cluster[layer_idx] for cluster in cluster_models)
            final_weights.append(layer_sum / len(cluster_models))

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
