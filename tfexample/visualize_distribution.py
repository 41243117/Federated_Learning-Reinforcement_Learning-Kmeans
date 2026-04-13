# visualize_distribution.py
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions
import matplotlib.pyplot as plt

# 1. 複製你在 task.py 中設定的 DirichletPartitioner 參數
partitioner = DirichletPartitioner(
    num_partitions=10,      # 假設你有 10 個 clients，請根據你的實際設定調整
    partition_by="label",
    alpha=0.1,              # 越小越不平均
    min_partition_size=10,  
    self_balancing=False,   
)

# 2. 載入資料集並套用 Partitioner
fds = FederatedDataset(
    dataset="uoft-cs/cifar10",
    partitioners={"train": partitioner},
)

# 3. 使用 Flower 官方提供的視覺化函式
fig, ax, df = plot_label_distributions(
    partitioner=fds.partitioners["train"],
    label_name="label",
    plot_type="bar", #bar        # 也可以改成 "heatmap" 來顯示熱力圖
    size_unit="percent",    # 將數量正規化為百分比，這樣每一條加總會是 100%
    partition_id_axis="x",  # 將 X 軸設為 Partition ID
    legend=True,            # 顯示標籤圖例
    verbose_labels=True,    # 顯示有意義的標籤名稱
    title="Client Label Distribution (Dirichlet alpha=0.1)",
)

# 顯示並儲存圖片
plt.tight_layout()
plt.savefig("label_distribution.png")
print("已將資料分佈圖儲存為 label_distribution.png")
plt.show()
