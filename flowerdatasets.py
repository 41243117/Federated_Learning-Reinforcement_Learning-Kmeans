import matplotlib.pyplot as plt
import os
os.environ["HF_TOKEN"] = "your Access Tokens"
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions


fds = FederatedDataset(
    dataset="cifar10",
    partitioners={
        "train": DirichletPartitioner(
            num_partitions=10,
            partition_by="label",
            alpha=0.3,
            seed=42,
            min_partition_size=0,
        ),
    },
)

partitioner = fds.partitioners["train"]


fig, ax, df = plot_label_distributions(
    partitioner,
    label_name="label",
    plot_type="heatmap",
    size_unit="absolute",
    partition_id_axis="x",
    legend=True,
    verbose_labels=True,
    title="Per Partition Labels Distribution",
    plot_kwargs={"annot": True},
)

fig.savefig("my_distribution_plot.png", bbox_inches="tight")
print("圖表已成功儲存為 my_distribution_plot.png")
