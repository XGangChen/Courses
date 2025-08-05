# Frozen_Lake_dql.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque

# Import our custom environment and wrapper
from FrozenLake_custom_env import CustomFrozenLakePolarBear, PartialObservationWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch version:", torch.__version__)
if torch.cuda.is_available():
    print("CUDA Available: ", torch.cuda.is_available())
    print("Device Count: ", torch.cuda.device_count())
    print("Current Device: ", torch.cuda.current_device())
    print("Device Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print('cpu')


class DQN(nn.Module):
    """ DQN network for partial obs (3x3 => 9 inputs if window_size=1) """
    def __init__(self, in_shape=(3,3), out_actions=4):
        super().__init__()
        # flatten (3,3)=9
        self.input_size = in_shape[0] * in_shape[1]
        hidden = 32

        self.fc1 = nn.Linear(self.input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, out_actions)

    def forward(self, x):
        # x shape: (batch, 3, 3) or (batch, 9)
        if len(x.shape) == 3:
            # flatten last two dims
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


class ReplayMemory():
    def __init__(self, maxlen=10000):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class FrozenLakeDQL():
    def __init__(self):
        # hyperparameters
        self.gamma = 0.95
        self.lr = 0.001
        self.sync_rate = 50
        self.batch_size = 32
        self.replay_size = 10000

        self.loss_fn = nn.MSELoss()

        # NEW: Keep a history of loss values
        self.loss_history = []

    def make_env(self, window_size=1, max_steps=100, is_slippery=True, render_mode=None): 
        """Create the custom environment + partial observation wrapper."""
        env = CustomFrozenLakePolarBear(
            render_mode=render_mode,
            desc=None,
            map_name=None,
            is_slippery=is_slippery,
            max_steps=max_steps,
        )
        env = PartialObservationWrapper(env, window_size=window_size)
        return env

    def train(self, episodes=1000, render=False):
        # Create environment  
        env = self.make_env(
            window_size=1, 
            max_steps=100, 
            is_slippery=False, 
            render_mode='human' if render else None
        )
        obs_shape = env.observation_space.shape  # (3,3)
        num_actions = env.action_space.n

        # DQN networks
        policy_dqn = DQN(in_shape=obs_shape, out_actions=num_actions).to(device)
        target_dqn = DQN(in_shape=obs_shape, out_actions=num_actions).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.lr)
        memory = ReplayMemory(self.replay_size)

        epsilon = 1.0
        epsilon_decay = 0.99
        epsilon_min = 0.01

        step_count = 0
        rewards_per_episode = []

        for ep in range(episodes):
            obs, _ = env.reset()
            # flatten obs => shape (9,)
            state = torch.tensor(obs, dtype=torch.float32).flatten().to(device)

            done = False
            truncated = False
            total_reward = 0.0

            while not done and not truncated:
                with torch.no_grad():
                    qvals = policy_dqn(state.unsqueeze(0))
                    best_action = qvals.argmax(dim=1).item()
                # Epsilon-like behavior (here replaced with 0.8 vs random)
                #   or you can revert to if random.random() < epsilon:
                if random.random() < 0.8:
                    action = best_action
                else:
                    action_space = list(range(num_actions))
                    action_space.remove(best_action)
                    action = random.choice(action_space)

                next_obs, reward, done, truncated, _ = env.step(action)
                next_state = torch.tensor(next_obs, dtype=torch.float32).flatten().to(device)

                # Store in replay
                memory.append((state, action, reward, next_state, done))

                # Move on
                state = next_state
                total_reward += reward
                step_count += 1

                # train if enough memory
                if len(memory) >= self.batch_size:
                    self.optimize(memory, policy_dqn, target_dqn, optimizer)

                # sync
                if step_count % self.sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())

            rewards_per_episode.append(total_reward)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        env.close()

        # Save policy
        torch.save(policy_dqn.state_dict(), "frozen_lake_partial_dql.pt")

        # ----- Plot results -----
        plt.figure(figsize=(16, 5))

        # Plot rewards
        plt.subplot(1,3,1)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.plot(rewards_per_episode)

        # Plot epsilon decay
        plt.subplot(1,3,2)
        plt.title("Epsilon Decay")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        x_vals = np.arange(len(rewards_per_episode))
        epsilons = [max(epsilon_min, 1.0 * (epsilon_decay**i)) for i in range(len(rewards_per_episode))]
        plt.plot(x_vals, epsilons, color='orange')

        # NEW: Plot loss history
        plt.subplot(1,3,3)
        plt.title("Loss History")
        plt.xlabel("Optimization Step")
        plt.ylabel("Loss")
        plt.plot(self.loss_history, color='red')

        plt.tight_layout()
        plt.show()

    def optimize(self, memory, policy_dqn, target_dqn, optimizer):
        batch = memory.sample(self.batch_size)
        # Each item: (state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(device)          # (batch, 9)
        actions = torch.LongTensor(actions).to(device)   # (batch,)
        rewards = torch.FloatTensor(rewards).to(device)  # (batch,)
        next_states = torch.stack(next_states).to(device)# (batch, 9)
        dones = torch.BoolTensor(dones).to(device)

        # Current Q
        qvals = policy_dqn(states)  # shape (batch_size, num_actions)
        # gather the Q-value for the chosen action
        current_q = qvals.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q from target net
        with torch.no_grad():
            next_qvals = target_dqn(next_states)  # shape (batch_size, num_actions)
            max_next_q = torch.max(next_qvals, dim=1)[0]  # shape (batch_size,)

        # If done, Q-target = reward, else reward + gamma * max_next_q
        target_q = rewards + self.gamma * max_next_q * (~dones)

        loss = self.loss_fn(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # NEW: Store the loss as a float (move to CPU first)
        self.loss_history.append(loss.detach().cpu().item())

    def test(self, episodes=30, render=True): 
        env = self.make_env(window_size=1, max_steps=50, is_slippery=False, 
                            render_mode='human' if render else None)
        obs_shape = env.observation_space.shape
        num_actions = env.action_space.n

        # Load policy
        policy_dqn = DQN(in_shape=obs_shape, out_actions=num_actions).to(device)
        policy_dqn.load_state_dict(torch.load("frozen_lake_partial_dql.pt"))
        policy_dqn.eval()

        for ep in range(episodes):
            obs, _ = env.reset()
            state = torch.tensor(obs, dtype=torch.float32).flatten().to(device)
            done = False
            truncated = False
            total_reward = 0.0

            while not done and not truncated:
                if render:
                    env.render()
                with torch.no_grad():
                    qvals = policy_dqn(state.unsqueeze(0))
                    best_action = qvals.argmax(dim=1).item()

                # Same "epsilon-like" or 0.8 threshold approach
                if random.random() < 0.8:
                    action = best_action
                else:
                    action_space = list(range(num_actions))
                    action_space.remove(best_action)
                    action = random.choice(action_space)

                next_obs, reward, done, truncated, _ = env.step(action)
                state = torch.tensor(next_obs, dtype=torch.float32).flatten().to(device)
                total_reward += reward

            print(f"Episode {ep+1}, total reward = {total_reward}")

        env.close()


if __name__ == "__main__":
    dql = FrozenLakeDQL()
    dql.train(episodes=1000, render=False)
    dql.test(episodes=30, render=True)
