import os
import random
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
ENV_NAME = 'CartPole-v1'
MAX_EPISODES = 300
MAX_STEPS = 500
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 1e-3
BUFFER_SIZE = 100_000
MIN_REPLAY_SIZE = 1_000
TARGET_UPDATE_EVERY = 10
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
SOLVED_SCORE = 475
SOLVED_WINDOW = 20
OUTPUT_DIR = 'output'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'inverted_pendulum_dqn_model.keras')


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def build_q_network(state_dim, action_dim):
    model = keras.Sequential([
        layers.Input(shape=(state_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(action_dim, activation='linear')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse'
    )
    return model


def epsilon_greedy_action(model, state, epsilon, action_dim):
    if np.random.rand() < epsilon:
        return np.random.randint(action_dim)
    q_values = model.predict(state[np.newaxis, :], verbose=0)[0]
    return int(np.argmax(q_values))


def train_step(q_network, target_network, replay_buffer, batch_size):
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    current_q = q_network.predict(states, verbose=0)
    next_q = target_network.predict(next_states, verbose=0)
    max_next_q = np.max(next_q, axis=1)

    targets = current_q.copy()
    for i in range(batch_size):
        target_value = rewards[i] if dones[i] else rewards[i] + GAMMA * max_next_q[i]
        targets[i, actions[i]] = target_value

    history = q_network.fit(states, targets, epochs=1, verbose=0)
    return float(history.history['loss'][0])


def evaluate_agent(model, env_name=ENV_NAME, episodes=5):
    env = gym.make(env_name)
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset(seed=SEED + 100 + ep)
        done = False
        truncated = False
        total_reward = 0
        while not (done or truncated):
            q_values = model.predict(state[np.newaxis, :], verbose=0)[0]
            action = int(np.argmax(q_values))
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
    env.close()
    return float(np.mean(rewards)), rewards


def plot_training(reward_history, avg_history, loss_history):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(reward_history, label='Episode Reward')
    plt.plot(avg_history, label=f'{SOLVED_WINDOW}-Episode Moving Average', linewidth=2)
    plt.axhline(SOLVED_SCORE, color='red', linestyle='--', label='Solved Threshold')
    plt.title('DQN Training Rewards on CartPole-v1')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dqn_training_rewards.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(loss_history, color='purple', label='Training Loss')
    plt.title('DQN Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dqn_training_loss.png'), dpi=200)
    plt.close()


def save_metrics(reward_history, avg_history, eval_mean, eval_rewards):
    import pandas as pd

    rewards_df = pd.DataFrame({
        'episode': np.arange(1, len(reward_history) + 1),
        'reward': reward_history,
        'moving_average_20': avg_history
    })
    rewards_df.to_csv(os.path.join(OUTPUT_DIR, 'dqn_training_rewards.csv'), index=False)

    eval_df = pd.DataFrame({
        'evaluation_episode': np.arange(1, len(eval_rewards) + 1),
        'reward': eval_rewards
    })
    eval_df.loc[len(eval_df)] = ['mean', eval_mean]
    eval_df.to_csv(os.path.join(OUTPUT_DIR, 'dqn_evaluation_rewards.csv'), index=False)


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f'Environment: {ENV_NAME}')
    print(f'State dimension: {state_dim}')
    print(f'Action dimension: {action_dim} (0=left, 1=right)')

    q_network = build_q_network(state_dim, action_dim)
    target_network = build_q_network(state_dim, action_dim)
    target_network.set_weights(q_network.get_weights())

    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    state, _ = env.reset(seed=SEED)
    for _ in range(MIN_REPLAY_SIZE):
        action = env.action_space.sample()
        next_state, reward, done, truncated, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done or truncated)
        state = next_state
        if done or truncated:
            state, _ = env.reset()

    epsilon = EPSILON_START
    reward_history = []
    avg_history = []
    loss_history = []

    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0

        for step in range(MAX_STEPS):
            action = epsilon_greedy_action(q_network, state, epsilon, action_dim)
            next_state, reward, done, truncated, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done or truncated)
            state = next_state
            episode_reward += reward

            loss = train_step(q_network, target_network, replay_buffer, BATCH_SIZE)
            loss_history.append(loss)

            if done or truncated:
                break

        reward_history.append(episode_reward)
        moving_avg = np.mean(reward_history[-SOLVED_WINDOW:])
        avg_history.append(moving_avg)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % TARGET_UPDATE_EVERY == 0:
            target_network.set_weights(q_network.get_weights())

        print(f'Episode {episode:03d} | Reward: {episode_reward:6.1f} | Avg({SOLVED_WINDOW}): {moving_avg:6.2f} | Epsilon: {epsilon:.3f}')

        if episode >= SOLVED_WINDOW and moving_avg >= SOLVED_SCORE:
            print(f'Environment solved in {episode} episodes with moving average reward {moving_avg:.2f}.')
            break

    q_network.save(MODEL_PATH)
    plot_training(reward_history, avg_history, loss_history)

    eval_mean, eval_rewards = evaluate_agent(q_network)
    save_metrics(reward_history, avg_history, eval_mean, eval_rewards)

    print('\nEvaluation rewards:', eval_rewards)
    print(f'Mean evaluation reward: {eval_mean:.2f}')
    print(f'Model saved to: {MODEL_PATH}')
    print('Artifacts saved in output/.')

    env.close()


if __name__ == '__main__':
    main()
