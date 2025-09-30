import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dqn_cartpole import DQNAgent
from tensorflow.keras.models import load_model

# Inisialisasi environment
env = gym.make('CartPole-v1', render_mode='rgb_array')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Load model yang telah dilatih
agent = DQNAgent(state_size, action_size)
try:
    agent.model = load_model("dqn_cartpole.keras")
    print("Model loaded successfully.")
except:
    print("Failed to load model, using untrained agent.")

agent.epsilon = 0  # Minimal eksplorasi saat pengujian

def visualize_episode():
    frames = []
    total_reward = 0
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    
    for time in range(500):
        frame = env.render()
        frames.append(frame)  # Simpan frame untuk animasi
        
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        state = np.reshape(next_state, [1, state_size])
        
        if done:
            print(f"Test Episode Score: {total_reward:.2f}")
            break
    
    return frames, total_reward

def create_animation(frames):
    fig, ax = plt.subplots()
    ax.axis('off')
    img = ax.imshow(frames[0])
    
    def update(frame):
        img.set_array(frame)
        return img,
    
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
    plt.show()

# Jalankan beberapa episode untuk analisis performa
num_tests = 5
scores = []
for i in range(num_tests):
    print(f"Running test episode {i+1}...")
    frames, score = visualize_episode()
    scores.append(score)

print(f"Average Test Score: {np.mean(scores):.2f}")

create_animation(frames)
env.close()
