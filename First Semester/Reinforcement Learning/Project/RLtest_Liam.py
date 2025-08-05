import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F

# import your new environment
from RL_test import FrozenLakeWithBearEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # Ensure computations use CPU
print("PyTorch version:", torch.__version__)
if torch.cuda.is_available():
    print("CUDA Available: ", torch.cuda.is_available())
    print("Device Count: ", torch.cuda.device_count())
    print("Current Device: ", torch.cuda.current_device())
    print("Device Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print('cpu')

# Define model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions) # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# FrozenLakeWithBear Deep Q-Learning
class FrozenLakeDQL():
    # Hyperparameters (adjustable)
    learning_rate_a = 0.001         # learning rate (alpha)
    discount_factor_g = 0.9         # discount rate (gamma)    
    network_sync_rate = 50          # how many steps before syncing the policy and target network
    replay_memory_size = 1000       # size of replay memory
    mini_batch_size = 32            # mini-batch size for training

    # Neural Network
    loss_fn = nn.MSELoss()
    optimizer = None

    ACTIONS = ['U','R','D','L']     # We'll match 0=up,1=right,2=down,3=left

    def train(self, episodes, render=False):
        # Create environment
        env = FrozenLakeWithBearEnv(
            render_mode='human' if render else None,
            map_size=4,
            hole_count=3,
            partial_obs_window=1,
            bear_chance=0.25,
            negative_reward=-1.0,
            goal_reward=1.0
        )

        # Because it's partial observation, let's define how big it is:
        #   partial_obs_window=1 => shape is (3,3) => 9.
        obs_shape = env.observation_space.shape  # (3,3)
        num_states = obs_shape[0] * obs_shape[1] # 9
        num_actions = env.action_space.n         # 4

        # Create policy and target network
        policy_dqn = DQN(in_states=num_states, h1_nodes=16, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=16, out_actions=num_actions)

        # Copy weights
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Initialize memory and optimizer
        memory = ReplayMemory(self.replay_memory_size)
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        epsilon = 1.0
        step_count = 0

        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []

        for i in range(episodes):
            obs, _ = env.reset()
            # obs shape=(3,3). Flatten => length 9
            state = obs.flatten()  # numpy array
            terminated = False
            truncated = False
            total_reward = 0

            while not terminated and not truncated:
                # Epsilon-greedy
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        # convert state to torch tensor
                        st_tensor = torch.FloatTensor(state)
                        qvals = policy_dqn(st_tensor)
                        action = qvals.argmax().item()

                # Step
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = next_obs.flatten()
                total_reward += reward

                # Save to memory
                memory.append((state, action, next_state, reward, terminated))

                # Move on
                state = next_state
                step_count += 1

                # Train if memory is sufficient
                if len(memory) > self.mini_batch_size:
                    self.optimize(memory, policy_dqn, target_dqn, num_states)

                # Sync policy -> target
                if step_count % self.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())

            rewards_per_episode[i] = total_reward
            # Decay epsilon
            epsilon = max(0.01, epsilon * 0.995)
            epsilon_history.append(epsilon)

        # Save policy
        torch.save(policy_dqn.state_dict(), "frozen_lake_with_bear_dql.pt")

        # Plot results
        plt.figure()
        plt.subplot(1,2,1)
        plt.title("Episode Rewards")
        plt.plot(rewards_per_episode)

        plt.subplot(1,2,2)
        plt.title("Epsilon Decay")
        plt.plot(epsilon_history)

        plt.show()
        env.close()

    def optimize(self, memory, policy_dqn, target_dqn, num_states):
        mini_batch = memory.sample(self.mini_batch_size)

        # separate columns
        states, actions, next_states, rewards, dones = zip(*mini_batch)

        states = torch.FloatTensor(states)         # shape: (batch, num_states)
        actions = torch.LongTensor(actions).view(-1,1)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards).view(-1,1)
        dones = torch.BoolTensor(dones).view(-1,1)

        # current Q
        current_q_vals = policy_dqn(states)
        current_q = current_q_vals.gather(1, actions)  # Q(s,a)

        # target Q
        with torch.no_grad():
            next_q_vals = target_dqn(next_states)
            max_next_q = next_q_vals.max(dim=1, keepdim=True)[0]
            target_q = rewards + self.discount_factor_g * max_next_q * (~dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, episodes=5, render=True):
        env = FrozenLakeWithBearEnv(
            render_mode='human' if render else None,
            map_size=4,
            hole_count=3,
            partial_obs_window=1,
            bear_chance=0.25,
            negative_reward=-1.0,
            goal_reward=1.0
        )
        obs_shape = env.observation_space.shape
        num_states = obs_shape[0]*obs_shape[1]
        num_actions = env.action_space.n

        # load policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=16, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("frozen_lake_with_bear_dql.pt"))
        policy_dqn.eval()

        for ep in range(episodes):
            obs, _ = env.reset()
            state = obs.flatten()
            terminated = False
            truncated = False
            total_reward = 0

            while not terminated and not truncated:
                if render:
                    env.render()

                st_tensor = torch.FloatTensor(state)
                with torch.no_grad():
                    qvals = policy_dqn(st_tensor)
                    action = qvals.argmax().item()

                next_obs, reward, terminated, truncated, _ = env.step(action)
                state = next_obs.flatten()
                total_reward += reward

            print(f"Episode {ep+1}: total reward = {total_reward}")

        env.close()

if __name__ == "__main__":
    dql = FrozenLakeDQL()
    dql.train(episodes=500, render=False)
    dql.test(episodes=5, render=True)
