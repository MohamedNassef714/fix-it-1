Public Class Form1
    pip install numpy gym tensorflow matplotlib
import numpy as np
import gym
import random
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
def build_model(state_size, action_size):
    model = tf.keras.Sequential([
        layers.Dense(24, input_dim=state_size, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model
def build_model(state_size, action_size):
    model = tf.keras.Sequential([
        layers.Dense(24, input_dim=state_size, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model
    Class ReplayBuffer
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size

    def add(self, experience):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)
def train_dqn(env, episodes=1000, batch_size=64, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, gamma=0.99):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    model = build_model(state_size, action_size)
    target_model = build_model(state_size, action_size)
    target_model.set_weights(model.get_weights())

    replay_buffer = ReplayBuffer(max_size=100000)

    rewards_list = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()  # Exploration
            else:
                q_values = model.predict(np.array([state]))
                action = np.argmax(q_values[0])  # Exploitation

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            replay_buffer.add((state, action, reward, next_state, done))

            if replay_buffer.size() > batch_size:
                minibatch = replay_buffer.sample(batch_size)
                for state_b, action_b, reward_b, next_state_b, done_b in minibatch:
                    target = reward_b
                    if not done_b:
                        target += gamma * np.max(target_model.predict(np.array([next_state_b]))[0])
                    target_q_values = model.predict(np.array([state_b]))
                    target_q_values[0][action_b] = target
                    model.fit(np.array([state_b]), target_q_values, epochs=1, verbose=0)

            state = next_state

        target_model.set_weights(model.get_weights())  # Update target model

        rewards_list.append(total_reward)
        if episode % 100 == 0:
            print(f"Episode {episode}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon}")

    return model, rewards_list
def test_dqn(env, model):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        q_values = model.predict(np.array([state]))
        action = np.argmax(q_values[0])
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        env.render()
    print(f"Test Total Reward: {total_reward}")
if __name__ == "__main__":
    env = gym.make("Taxi-v3")

    # Train the model
    print("Training the model...")
    trained_model, rewards = train_dqn(env)

    # Plot rewards over time
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()

    # Test the trained model
    print("Testing the model...")
    test_dqn(env, trained_model)

        Private Sub Form1_Load(sender As System.Object, e As System.EventArgs) Handles MyBase.Load

        End Sub
    End Class
