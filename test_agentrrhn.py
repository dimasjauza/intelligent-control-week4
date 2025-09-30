import gym
import numpy as np
from dqn_agent import DQNAgent

# Tambahkan render_mode agar simulasi tampil
env = gym.make('CartPole-v1', render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Inisialisasi agen dengan arsitektur sama
agent = DQNAgent(state_size, action_size)

# Load bobot hasil training
agent.model.load_weights("dqn_cartpole.weights.h5")
agent.epsilon = 0.01  # minim eksplorasi saat testing

for e in range(5):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])

    for time in range(500):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = np.reshape(next_state, [1, state_size])
        if done:
            print(f"Test Episode: {e+1}, Score: {time}")
            break

env.close()
