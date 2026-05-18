"""tfexample: A Flower / TensorFlow app."""

import os
import random
import keras
import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.partitioner import IidPartitioner
from keras import layers

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model(learning_rate: float = 0.001):
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    #optimizer = keras.optimizers.Adam(learning_rate)
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

fds = None  
partition_cache = {}  
malicious_clients = None

def load_data(partition_id, num_partitions):
    global fds, partition_cache, malicious_clients

    if partition_id in partition_cache:
        return partition_cache[partition_id]

    if fds is None:
    
        import random
        import os
        
        my_seed = int(os.environ.get("SIMULATION_SEED", 42))
        random.seed(my_seed)
        
        malicious_clients = random.sample(range(num_partitions), 1)
        print(f"\n🎲 [系統公告] 抽籤完畢！本次實驗的隨機惡意節點為: Client {malicious_clients} 🎲\n")
       
        #from flwr_datasets.partitioner import IidPartitioner
        partitioner = IidPartitioner(num_partitions=num_partitions)
       
        #partitioner = DirichletPartitioner(
        #    num_partitions=num_partitions,
        #    partition_by="label",
        #    alpha=0.1,              
        #    min_partition_size=10,  
        #    self_balancing=False,   
        #)
        fds = FederatedDataset(
            dataset="mnist",
            partitioners={"train": partitioner},
        )
        #fds = FederatedDataset(
        #    dataset="uoft-cs/cifar10",
        #    partitioners={"train": partitioner},
        #)
    partition = fds.load_partition(partition_id, "train")
    partition = partition.train_test_split(test_size=0.2)

    partition["train"].set_format(type="numpy", columns=["image", "label"])
    partition["test"].set_format(type="numpy", columns=["image", "label"])

    x_train = partition["train"][:]["image"].astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    y_train = partition["train"][:]["label"]
    x_test = partition["test"][:]["image"].astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1)
    y_test = partition["test"][:]["label"]
    partition["train"].set_format(type="numpy", columns=["img", "label"])
    partition["test"].set_format(type="numpy", columns=["img", "label"])

    #x_train = partition["train"][:]["img"].astype("float32") / 255.0
    #y_train = partition["train"][:]["label"]
    #x_test = partition["test"][:]["img"].astype("float32") / 255.0
    #y_test = partition["test"][:]["label"]
    # ==========================================
    # 😈 惡意節點標籤翻轉 (Label Flipping) 邏輯 😈
    # ==========================================
    # 假設我們設定總共 10 個 Client 中，前 2 個 (ID: 0, 1) 是惡意的
    # malicious_clients = [0, 1] 
    
    if partition_id in malicious_clients:
        print(f"\n🚨 [警告] Client {partition_id} 啟動惡意模式，正在污染訓練標籤！ 🚨")
        
        # 這裡提供三種常見的翻轉策略，你可以把想用的「取消註解」：

        # 🎯 策略 1：對稱翻轉 (Symmetric Flipping) - 最常見
        # 把標籤 0 變成 9, 1 變成 8... 整個徹底搞亂
        #y_train = 9 - y_train

        # 🎯 策略 2：目標翻轉 (Targeted Flipping) 
        # 把所有的飛機 (0) 強制標記成鳥 (2)，針對性攻擊
        y_train = np.where(y_train == 1, 7, y_train)

        # 🎯 策略 3：完全隨機打亂 (Random Shuffling)
        # 標籤變成毫無意義的雜訊
        # np.random.shuffle(y_train)
    # ==========================================

    partition_cache[partition_id] = (x_train, y_train, x_test, y_test)

    return partition_cache[partition_id]
