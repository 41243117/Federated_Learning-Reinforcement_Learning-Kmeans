"""tfexample: A Flower / TensorFlow app."""

import os
import random

import keras
import numpy as np
from keras import layers


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# =========================
# Experiment Settings
# =========================

SEED = int(os.environ.get("SIMULATION_SEED", 42))

# 控制 Non-IID client 比例
# 例如：
# 0.0 = 全部 IID
# 0.2 = 20% Non-IID
# 0.4 = 40% Non-IID
# 1.0 = 全部 Non-IID
NONIID_RATIO = float(os.environ.get("NONIID_RATIO", 0.0))

# Non-IID client 中，主要標籤所佔比例
# 越高代表越偏斜
NONIID_PRIMARY_RATIO = float(os.environ.get("NONIID_PRIMARY_RATIO", 0.8))

# 是否啟用標籤翻轉攻擊
# 0 = 關閉攻擊
# 1 = 開啟攻擊
ENABLE_ATTACK = os.environ.get("ENABLE_ATTACK", "1") == "1"

# 標籤翻轉攻擊：把 1 翻成 7
ATTACK_FROM_LABEL = 1
ATTACK_TO_LABEL = 7

# 惡意節點數量
NUM_MALICIOUS_CLIENTS = int(os.environ.get("NUM_MALICIOUS_CLIENTS", 1))


# =========================
# Global Cache
# =========================

partition_cache = {}

data_initialized = False
x_train_full = None
y_train_full = None
x_test_full = None
y_test_full = None

train_indices_by_client = {}
test_indices_by_client = {}

iid_clients = []
noniid_clients = []
malicious_clients = []


# =========================
# Model
# =========================

def load_model(learning_rate: float = 0.001):
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

    optimizer = keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=0.9,
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =========================
# Partition Helpers
# =========================

def _make_label_pools(labels, seed):
    """依照 label 建立索引池，並固定亂數順序。"""
    rng = np.random.default_rng(seed)
    pools = {}

    for label in range(10):
        idx = np.where(labels == label)[0].astype(np.int64)
        rng.shuffle(idx)
        pools[label] = idx.tolist()

    return pools


def _take_from_label(pools, label, amount):
    """從指定 label 的資料池取出 amount 筆。"""
    amount = int(amount)
    available = len(pools[label])
    take_amount = min(amount, available)

    selected = pools[label][:take_amount]
    pools[label] = pools[label][take_amount:]

    return selected


def _take_any(pools, amount):
    """當指定 label 不足時，從其他 label 補資料。"""
    selected = []
    amount = int(amount)

    while len(selected) < amount:
        available_labels = [label for label in range(10) if len(pools[label]) > 0]

        if not available_labels:
            break

        for label in available_labels:
            if len(selected) >= amount:
                break

            selected.extend(_take_from_label(pools, label, 1))

    return selected


def _split_indices_hybrid(labels, num_partitions, noniid_client_ids, iid_client_ids, seed):
    """
    建立混合式資料切分：
    - Non-IID clients：資料集中在少數幾個 label
    - IID clients：從剩下資料中盡量平均分配各 label
    """
    rng = np.random.default_rng(seed)
    pools = _make_label_pools(labels, seed)

    client_indices = {cid: [] for cid in range(num_partitions)}
    target_size = len(labels) // num_partitions

    # -------------------------
    # 1. 先切 Non-IID clients
    # -------------------------
    for cid in noniid_client_ids:
        primary_label = cid % 10
        secondary_label = (cid + 1) % 10

        primary_amount = int(target_size * NONIID_PRIMARY_RATIO)

        selected = []
        selected.extend(_take_from_label(pools, primary_label, primary_amount))

        remaining_amount = target_size - len(selected)
        selected.extend(_take_from_label(pools, secondary_label, remaining_amount))

        if len(selected) < target_size:
            selected.extend(_take_any(pools, target_size - len(selected)))

        rng.shuffle(selected)
        client_indices[cid].extend(selected)

    # -------------------------
    # 2. 再切 IID clients
    # -------------------------
    if len(iid_client_ids) > 0:
        for label in range(10):
            label_indices = np.array(pools[label], dtype=np.int64)
            rng.shuffle(label_indices)

            chunks = np.array_split(label_indices, len(iid_client_ids))

            for cid, chunk in zip(iid_client_ids, chunks):
                client_indices[cid].extend(chunk.tolist())

    # -------------------------
    # 3. 如果全部都是 Non-IID，把剩下資料補回 Non-IID clients
    # -------------------------
    else:
        remaining = []

        for label in range(10):
            remaining.extend(pools[label])

        rng.shuffle(remaining)

        if len(noniid_client_ids) > 0:
            chunks = np.array_split(np.array(remaining, dtype=np.int64), len(noniid_client_ids))

            for cid, chunk in zip(noniid_client_ids, chunks):
                client_indices[cid].extend(chunk.tolist())

    # -------------------------
    # 4. 每個 client 內部資料順序打亂
    # -------------------------
    for cid in range(num_partitions):
        rng.shuffle(client_indices[cid])

    return client_indices


def _initialize_data(num_partitions):
    """只在第一次呼叫時建立所有 client 的資料分配。"""
    global data_initialized
    global x_train_full, y_train_full, x_test_full, y_test_full
    global train_indices_by_client, test_indices_by_client
    global iid_clients, noniid_clients, malicious_clients

    if data_initialized:
        return

    random.seed(SEED)
    np.random.seed(SEED)

    # 載入 MNIST
    (x_train_full, y_train_full), (x_test_full, y_test_full) = keras.datasets.mnist.load_data()

    y_train_full = y_train_full.astype(np.int64)
    y_test_full = y_test_full.astype(np.int64)

    # 決定 Non-IID client 數量
    num_noniid = int(round(num_partitions * NONIID_RATIO))
    num_noniid = max(0, min(num_noniid, num_partitions))

    # 為了讓 0%、20%、40% 之間比較穩定，這裡固定從前面的 client 開始設成 Non-IID
    noniid_clients = list(range(num_noniid))
    iid_clients = [cid for cid in range(num_partitions) if cid not in noniid_clients]

    # 固定惡意 client
    num_malicious = min(NUM_MALICIOUS_CLIENTS, num_partitions)
    malicious_clients = list(range(num_malicious))

    # 建立 train / test 的混合 IID + Non-IID 切分
    train_indices_by_client = _split_indices_hybrid(
        labels=y_train_full,
        num_partitions=num_partitions,
        noniid_client_ids=noniid_clients,
        iid_client_ids=iid_clients,
        seed=SEED,
    )

    test_indices_by_client = _split_indices_hybrid(
        labels=y_test_full,
        num_partitions=num_partitions,
        noniid_client_ids=noniid_clients,
        iid_client_ids=iid_clients,
        seed=SEED + 1,
    )

    print("=" * 80)
    print("Hybrid IID / Non-IID Data Split")
    print(f"SEED = {SEED}")
    print(f"NONIID_RATIO = {NONIID_RATIO}")
    print(f"NONIID clients = {noniid_clients}")
    print(f"IID clients = {iid_clients}")
    print(f"ENABLE_ATTACK = {ENABLE_ATTACK}")
    print(f"Malicious clients = {malicious_clients}")
    print("=" * 80)

    data_initialized = True


# =========================
# Load Data
# =========================

def load_data(partition_id, num_partitions):
    global partition_cache

    partition_id = int(partition_id)
    num_partitions = int(num_partitions)

    # 每個 client 每輪都拿同一份資料
    if partition_id in partition_cache:
        return partition_cache[partition_id]

    _initialize_data(num_partitions)

    train_idx = train_indices_by_client[partition_id]
    test_idx = test_indices_by_client[partition_id]

    x_train = x_train_full[train_idx].astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    y_train = y_train_full[train_idx].copy()

    x_test = x_test_full[test_idx].astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1)
    y_test = y_test_full[test_idx].copy()

    # 只攻擊訓練資料，不攻擊測試資料
    if ENABLE_ATTACK and partition_id in malicious_clients:
        y_train = np.where(
            y_train == ATTACK_FROM_LABEL,
            ATTACK_TO_LABEL,
            y_train,
        )

    split_type = "Non-IID" if partition_id in noniid_clients else "IID"
    label_counts = np.bincount(y_train, minlength=10)

    print(
        f"[Client {partition_id}] "
        f"type={split_type}, "
        f"train={len(y_train)}, "
        f"test={len(y_test)}, "
        f"label_counts={label_counts.tolist()}"
    )

    partition_cache[partition_id] = (x_train, y_train, x_test, y_test)

    return partition_cache[partition_id]
