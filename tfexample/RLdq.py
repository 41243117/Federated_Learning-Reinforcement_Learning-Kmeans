import numpy as np
import tensorflow as tf

# --- Step 1: 建立 Keras 神經網路 ---
def build_net(n_states, n_actions, n_hidden):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(n_states,)),
        tf.keras.layers.Dense(n_hidden, activation='relu'),
        # 加入第二層隱藏層，讓 Agent 對於特徵的理解能力更好 (可選)
        tf.keras.layers.Dense(n_hidden, activation='relu'), 
        tf.keras.layers.Dense(n_actions, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    return model

# --- Step 2: 建立 DQN Agent 邏輯 ---
class DQN:
    # 這裡的預設值可以依據你的 FL 環境調整 (例如 memory_capacity 可能不需要 2000 這麼大，因為 FL 輪數通常不多)
    def __init__(self, n_states, n_actions, n_hidden=50, lr=0.01, gamma=0.9, epsilon=0.9, 
                 target_replace_iter=50, memory_capacity=500):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity

        # 初始化 Eval 與 Target 網路
        self.eval_net = build_net(n_states, n_actions, n_hidden)
        self.target_net = build_net(n_states, n_actions, n_hidden)
        
        # 初始時讓兩邊權重一致
        self.target_net.set_weights(self.eval_net.get_weights())

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2))

    def choose_action(self, state):
        # 將輸入狀態轉為二維，符合 Keras 預測格式
        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # 根據網路預測選擇 Q 值最大的動作
            actions_value = self.eval_net.predict(state, verbose=0)
            action = np.argmax(actions_value[0])
        else:
            # 探索：隨機選擇動作
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 記憶庫資料不夠時先不訓練 (例如設定至少要收集 32 筆)
        if self.memory_counter < 32:
            return

        # 1. 檢查是否需要更新 Target Network
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.set_weights(self.eval_net.get_weights())
        self.learn_step_counter += 1

        # 2. 從記憶庫中抽出 Batch
        sample_index = np.random.choice(min(self.memory_counter, self.memory_capacity), 32)
        b_memory = self.memory[sample_index, :]
        
        b_s = b_memory[:, :self.n_states]
        b_a = b_memory[:, self.n_states].astype(int)
        b_r = b_memory[:, self.n_states+1]
        b_s_ = b_memory[:, -self.n_states:]

        # 3. 計算 Q Target
        q_next = self.target_net.predict(b_s_, verbose=0)
        q_eval = self.eval_net.predict(b_s, verbose=0)
        
        q_target = q_eval.copy()
        batch_index = np.arange(32, dtype=np.int32)
        
        q_target[batch_index, b_a] = b_r + self.gamma * np.max(q_next, axis=1)

        # 4. 訓練網路
        self.eval_net.train_on_batch(b_s, q_target)

    # (可選) 幫你加一個儲存/載入模型的功能，因為 FL 訓練很久，萬一中斷可以接續
    def save_model(self, path="dqn_model.keras"):
        self.eval_net.save(path)
        
    def load_model(self, path="dqn_model.keras"):
        self.eval_net = tf.keras.models.load_model(path)
        self.target_net.set_weights(self.eval_net.get_weights())
