import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple, deque
from itertools import count
from PIL import Image

# Set up environment and device
env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Experience tuple and buffer
Experience = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ExperienceBuffer:
    def __init__(self, max_capacity):
        self.buffer = deque([], maxlen=max_capacity)

    def add(self, *args):
        self.buffer.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Define DQN model
class DQN(nn.Module):
    def __init__(self, input_dim, action_count):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_count)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500
TARGET_UPDATE = 10
LEARNING_RATE = 1e-3
BUFFER_SIZE = 10000
NUM_EPISODES = 500

# Initialize networks, buffer, and optimizer
n_actions = env.action_space.n
state_dim = env.observation_space.shape[0]
policy_net = DQN(state_dim, n_actions).to(device)
target_net = DQN(state_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
replay_buffer = ExperienceBuffer(BUFFER_SIZE)
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

# Epsilon-greedy action selection
steps_done = 0
def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

# Optimize the model
def optimize_model():
    if len(replay_buffer) < BATCH_SIZE:
        return
    transitions = replay_buffer.sample(BATCH_SIZE)
    batch = Experience(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# Training loop
episode_rewards = []
for i_episode in range(NUM_EPISODES):
    state = torch.tensor([env.reset()[0]], dtype=torch.float32, device=device)
    total_reward = 0

    for t in count():
        action = select_action(state)
        obs, reward, done, _, _ = env.step(action.item())
        total_reward += reward
        reward = torch.tensor([reward], device=device)
        next_state = torch.tensor([obs], dtype=torch.float32, device=device) if not done else None
        replay_buffer.add(state, action, next_state, reward)

        state = next_state
        optimize_model()
        if done:
            episode_rewards.append(total_reward)
            break

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Log progress
    if i_episode % 10 == 0:
        print(f"Episode {i_episode}/{NUM_EPISODES}, Reward: {total_reward}")

# Plot results
plt.figure()
plt.plot(episode_rewards, label='Total Reward')
plt.title('Training Progress')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()

# Demo function
def run_demo():
    for episode in range(5):
        state = torch.tensor([env.reset()[0]], dtype=torch.float32, device=device)
        total_reward = 0
        for t in count():
            env.render()
            action = policy_net(state).argmax(dim=1).view(1, 1)
            obs, reward, done, _, _ = env.step(action.item())
            total_reward += reward
            state = torch.tensor([obs], dtype=torch.float32, device=device) if not done else None
            if done:
                print(f"Demo Episode {episode + 1}: Total Reward = {total_reward}")
                break
    env.close()

print("Training completed!")
run_demo()
