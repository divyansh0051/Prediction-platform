"""Train a DQN agent using a simple trading environment.

This is a standalone DQN implementation using TensorFlow/Keras directly.
No keras-rl2 dependency needed.
"""
import os
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from envs.trading_env import TradingEnv


class SimpleMemory:
    """Simple replay memory for DQN."""
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.memory = []

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        return batch

    def __len__(self):
        return len(self.memory)


def build_model(input_shape, nb_actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
    return model


class DQNAgent:
    def __init__(self, model, nb_actions, memory, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.model = model
        self.target_model = tf.keras.models.clone_model(model)
        self.target_model.set_weights(model.get_weights())
        self.nb_actions = nb_actions
        self.memory = memory
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        self.memory.remember(state, action, reward, next_state, done)

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.nb_actions)
        else:
            q_vals = self.model.predict(state.reshape(1, 1, -1), verbose=0)
            return np.argmax(q_vals[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        try:
            batch = self.memory.sample(batch_size)
            states = np.array([x[0] for x in batch], dtype=np.float32)
            actions = np.array([x[1] for x in batch], dtype=np.int32)
            rewards = np.array([x[2] for x in batch], dtype=np.float32)
            next_states = np.array([x[3] for x in batch], dtype=np.float32)
            dones = np.array([x[4] for x in batch], dtype=np.float32)

            states = states.reshape(batch_size, 1, -1)
            next_states = next_states.reshape(batch_size, 1, -1)

            target_q_vals = self.target_model.predict(next_states, verbose=0)
            target_q_vals_train = self.model.predict(states, verbose=0)

            for i in range(batch_size):
                if dones[i]:
                    target_q_vals_train[i][actions[i]] = rewards[i]
                else:
                    target_q_vals_train[i][actions[i]] = rewards[i] + self.gamma * np.max(target_q_vals[i])

            self.model.fit(states, target_q_vals_train, epochs=1, verbose=0, batch_size=16)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        except Exception as e:
            print(f"Replay error: {e}")
            pass

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)
        self.update_target_model()


def train_dqn_on_prices(prices, nb_steps=50):
    # enable GPU memory growth if possible
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f"GPU enabled: {len(gpus)} device(s)")
    except Exception:
        pass

    env = TradingEnv(prices, window=10)
    np.random.seed(123)
    try:
        env.seed(123)
    except Exception:
        pass

    nb_actions = env.action_space.n
    model = build_model(env.observation_space.shape, nb_actions)
    memory = SimpleMemory(capacity=50000)
    agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory)

    print(f"Starting DQN training for {nb_steps} episodes...")
    for step in range(nb_steps):
        state = env.reset()
        done = False
        episode_reward = 0
        steps_in_episode = 0

        while not done and steps_in_episode < 100:  # limit steps per episode
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            # only replay after warmup
            if len(agent.memory) > 64:
                try:
                    agent.replay(batch_size=32)
                except Exception as e:
                    print(f"Replay error: {e}")
                    break
                    
            state = next_state
            episode_reward += reward
            steps_in_episode += 1

        if (step + 1) % max(1, nb_steps // 10) == 0:
            print(f"Episode {step + 1}/{nb_steps}, Epsilon: {agent.epsilon:.4f}, Reward: {episode_reward:.2f}, Memory: {len(agent.memory)}")
            agent.update_target_model()

    os.makedirs('models', exist_ok=True)
    agent.save_weights('models/dqn_weights.weights.h5')
    print('DQN training complete, weights saved to models/dqn_weights.weights.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DQN agent on price series')
    parser.add_argument('--nb-steps', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--ticker', '-t', help='Yahoo ticker to train on (e.g. HDFC.NS). '
                        'If omitted, a random walk is used.')
    args = parser.parse_args()

    if args.ticker:
        try:
            import yfinance as yf
            print(f"Fetching price data for {args.ticker}...")
            df = yf.download(args.ticker, period='720d', interval='1d', progress=False)
            if df.empty:
                raise ValueError('no data returned')
            prices = df['Close'].astype(float).tolist()
            print(f"Loaded {len(prices)} price points for training")
        except Exception as e:
            print(f"Error loading ticker {args.ticker}: {e}")
            prices = np.cumsum(np.random.randn(2000)) + 1000
            print('Falling back to random walk')
    else:
        # random walk default
        prices = np.cumsum(np.random.randn(2000)) + 1000

    train_dqn_on_prices(prices, nb_steps=args.nb_steps)
