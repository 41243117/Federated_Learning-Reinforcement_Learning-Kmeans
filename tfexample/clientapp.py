import numpy as np
import keras
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from tfexample.task import load_data, load_model

app = ClientApp()

# 多層權重攤平成一條向量
def flatten_weights(weights_list):
    return np.concatenate([w.reshape(-1) for w in weights_list]).astype(np.float64)
    # np.concatenate -> 所有層接起來
    # w.reshape(-1) -> 每層攤平成一維

def l2_norm_of_delta(global_w, local_w):
    deltas = [lw - gw for lw, gw in zip(local_w, global_w)]
    # zip -> 打包並逐層對齊
    flat = flatten_weights(deltas)
    return float(np.linalg.norm(flat))
    # np.linalg.norm() -> 算 L2 norm

def cosine_similarity_of_delta(global_w, local_w):
    delta = flatten_weights([lw - gw for lw, gw in zip(local_w, global_w)])
    ref = flatten_weights(global_w)

    denom = (np.linalg.norm(delta) * np.linalg.norm(ref))
    if denom == 0:
        return 0.0
    return float(np.dot(delta, ref) / denom)

def loss_on_data(model, x, y):
    out = model.evaluate(x, y, verbose=0)
    # model.evaluate() -> 回傳 loss 值或 accuracy 等指標
    if isinstance(out, (list, tuple)):
        return float(out[0])
    return float(out)

# 取出各 client 的資料與計算比率
def label_distribution(y, num_classes: int = 10):
    y = np.asarray(y).astype(np.int64)
    counts = np.bincount(y, minlength=num_classes).astype(int)
    total = int(counts.sum())
    ratios = (counts / total).astype(np.float64) if total > 0 else np.zeros(num_classes, dtype=np.float64)
    return counts, ratios

@app.train()
def train(msg: Message, context: Context):
    keras.backend.clear_session()

    # 載入 client 自己的資料
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    x_train, y_train, _, _ = load_data(partition_id, num_partitions)

    # 顯示每個 client 中各標籤的資料量與比率
    counts, ratios = label_distribution(y_train, num_classes=10)
    print(f"[Client {partition_id}] label_counts={counts.tolist()}")
    print(f"[Client {partition_id}] label_ratios={[float(r) for r in ratios.tolist()]}")
    
    # 建立模型
    lr = context.run_config["learning-rate"]
    model = load_model(lr)

    # 接收自 server 傳來的 global weights
    global_weights = msg.content["arrays"].to_numpy_ndarrays()
    model.set_weights(global_weights)

    # 讀取本地訓練超參數
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    loss_global = loss_on_data(model, x_train, y_train)

    # 本地訓練（標準 Keras 訓練）
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    local_weights = model.get_weights()

    loss_local = loss_on_data(model, x_train, y_train)

    # 計算三特徵
    cos_sim = cosine_similarity_of_delta(global_weights, local_weights)
    l2 = l2_norm_of_delta(global_weights, local_weights)
    loss = float(loss_local - loss_global)

    # 從 history 取出訓練指標
    train_loss = history.history["loss"][-1] if "loss" in history.history else None
    train_acc = history.history["accuracy"][-1] if "accuracy" in history.history else None

    # 封裝回傳模型與 metrics
    model_record = ArrayRecord(local_weights)
    metrics = {"num-examples": len(x_train)}

    # 加入 train_loss/train_acc
    if train_loss is not None:
        metrics["train_loss"] = float(train_loss)
    if train_acc is not None:
        metrics["train_acc"] = float(train_acc)

    # 給 KMeans 的三個值與 loss 資訊
    metrics["cos_sim"] = cos_sim
    metrics["l2_norm"] = l2
    metrics["loss"] = loss
    metrics["loss_global"] = loss_global
    metrics["loss_local"] = loss_local

    content = RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)

@app.evaluate() # client 的評估流程
def evaluate(msg: Message, context: Context):

    # 清除舊 session
    keras.backend.clear_session()

    # 載入測試資料
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, _, x_test, y_test = load_data(partition_id, num_partitions)

    # 建立模型並套上 server 給的權重
    lr = context.run_config["learning-rate"]
    model = load_model(lr)
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

    # 執行測試
    results = model.evaluate(x_test, y_test, verbose=0)

    # 統一處理輸出格式
    if isinstance(results, (list, tuple)):
        eval_loss = float(results[0])
        eval_acc = float(results[1]) if len(results) > 1 else 0.0
    else:
        eval_loss = float(results)
        eval_acc = 0.0

    # 封裝評估指標回傳
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(x_test),
    }

    # 回傳評估訊息
    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
