import numpy as np
import tensorflow as tf
import gymnasium as gym

# --- Step 1: 建立 Keras 神經網路 ---
def build_net(n_states, n_actions, n_hidden):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(n_states,)),
        tf.keras.layers.Dense(n_hidden, activation='relu'),
        tf.keras.layers.Dense(n_actions, activation='linear') # 輸出 Q 值不需要 activation
    ])
    # 使用 MSE Loss 與 Adam Optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    return model

# --- Step 2: 建立 DQN Agent 邏輯 (Keras 版) ---
class DQN:
    def __init__(self, n_states, n_actions, n_hidden, lr=0.01, gamma=0.9, epsilon=0.9, 
                 target_replace_iter=100, memory_capacity=2000):
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

    def choose_action(self, x):
        # Keras 預測需要 batch 維度，例如形狀變為 (1, n_states)
        x = x[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # 取得 Q 值並選擇最大值的動作 (verbose=0 關閉預測時的進度條)
            actions_value = self.eval_net.predict(x, verbose=0)
            action = np.argmax(actions_value[0])
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
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

        # 3. 計算 Q Target (使用 Bellman Equation)
        q_next = self.target_net.predict(b_s_, verbose=0)
        q_eval = self.eval_net.predict(b_s, verbose=0)
        
        q_target = q_eval.copy()
        batch_index = np.arange(32, dtype=np.int32)
        
        # 更新選擇的動作的 Q 值
        q_target[batch_index, b_a] = b_r + self.gamma * np.max(q_next, axis=1)

        # 4. 訓練網路 (使用 train_on_batch 進行單步更新)
        self.eval_net.train_on_batch(b_s, q_target)

# --- 將訓練邏輯包裝成函式，供後續呼叫 ---
def train_dqn(dqn_agent, env, num_episodes=5):
    """
    執行 RL 訓練並回傳平均獎勵
    """
    total_reward = 0
    for i_episode in range(num_episodes):
        s, _ = env.reset()
        ep_r = 0
        while True:
            a = dqn_agent.choose_action(s)
            s_, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            # 修改獎勵機制 (獎勵因子演練)
            x, x_dot, theta, theta_dot = s_
            r1 = (env.unwrapped.x_threshold - abs(x)) / env.unwrapped.x_threshold - 0.8
            r2 = (env.unwrapped.theta_threshold_radians - abs(theta)) / env.unwrapped.theta_threshold_radians - 0.5
            reward = r1 + r2

            dqn_agent.store_transition(s, a, reward, s_)
            ep_r += reward

            if dqn_agent.memory_counter > dqn_agent.memory_capacity:
                dqn_agent.learn()

            if done:
                print(f'Episode: {i_episode} | Reward: {round(ep_r, 2)}')
                break
            s = s_
        total_reward += ep_r
    return total_reward / num_episodes

# --- 單機測試區塊 ---
if __name__ == "__main__":
    # render_mode 若在 Server 執行請改為 None
    env = gym.make('CartPole-v1', render_mode="human")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    dqn = DQN(n_states=n_states, n_actions=n_actions, n_hidden=50)
    
    print("\n開始收集經驗並進行訓練...")
    train_dqn(dqn, env, num_episodes=50)
    
    print("測試完成！")
    env.close()
