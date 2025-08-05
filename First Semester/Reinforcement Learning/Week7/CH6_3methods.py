# Slide Example for Q-Learning (RL-Course NTNU, Saeedvand)
# (A) : Compare three algrithems:Q-learning, SARSA, E-SARSA
# (B) : Epsilon-greedy ~> ?

import gymnasium as gym
import numpy as np
import time
import math
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')#, render_mode="human") 
# env = gym.make('FrozenLake-v1')
# env = gym.make('CliffWalking-v0')#, render_mode="human") 
# env = gym.make('Myslide-v1')  # Important: The player may slip (20% error).
# env = gym.make("Blackjack-v1") # **Practice, Solve**

def plot(Qlearning_rewards, Qlearning_epsilon_decay, SARSA_rewards, SARSA_epsilon_decay, ESARSA_rewards, ESARSA_epsilon_decay):
    fig, ax1 = plt.subplots()

    color = 'tab:green'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color=color)
    ax1.plot(Qlearning_rewards, color='red', label='Qlearning_Reward')
    ax1.plot(SARSA_rewards, color='green', label='SARSA_Reward')
    ax1.plot(ESARSA_rewards, color='blue', label='ESARSA_Reward')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Epsilon', color=color)  # we already handled the x-label with ax1
    ax2.plot(Qlearning_epsilon_decay, color='red', label='Qlearning Epsilon Decay')
    ax2.plot(SARSA_epsilon_decay, color='green', label='SARSA Epsilon Decay')
    ax2.plot(ESARSA_epsilon_decay, color='blue', label='ESARSA Epsilon Decay')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Total Reward and Epsilon Decay - Q-Learning')
    plt.grid(axis='x', color='0.80')
    plt.show()


def Q_value_initialize(state, action, type = 0):
    if type == 1:
        return np.ones((state, action))
    elif type == 0:
        return np.zeros((state, action))
    elif type == -1:
        return np.random.random((state, action))
   

def epsilon_greedy(Q, epsilon, s):
    if np.random.rand() > epsilon:
        action = np.argmax(Q[s, :]).item()
    else:
        action = env.action_space.sample() 
    return action

def normalize(list): # you can use this to normalize your plot values
    xmin = min(list) 
    xmax = max(list)
    for i, x in enumerate(list):
        list[i] = (x-xmin) / (xmax-xmin)
    return list 

# ------------------------------------------------------------------------------------------------------------------------
def Qlearning(alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, n_tests):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = Q_value_initialize(n_states, n_actions, type = 0)
    timestep_reward = []
    epsilon_values = [] 
    epsilon_decay_rate = -np.log(EPS_END / EPS_START) / episodes
    
    for episode in range(episodes):
        print(f"Episode: {episode}")
        s, info = env.reset() # read also state
        
        epsilon_threshold = EPS_START * np.exp(-epsilon_decay_rate * episode)
        epsilon_values.append(epsilon_threshold)

        t = 0
        total_reward = 0
        while t < max_steps:
            t += 1
            a = epsilon_greedy(Q, epsilon_threshold, s)

            # Reward
            # -1 per step unless other reward is triggered.
            # +20 delivering passenger.
            # -10 executing “pickup” and “drop-off” actions illegally.
            s_, reward, terminated, truncated, info = env.step(a)

            #s_, reward, done, info = env.step(a)
            total_reward += reward
            a_max_action = np.argmax(Q[s_, :]).item()

            if terminated or truncated:
                Q[s, a] += alpha * (reward - Q[s, a])
            else:
                Q[s, a] += alpha * (reward + (gamma * Q[s_, a_max_action]) - Q[s, a])
            s, a = s_, a_max_action
            
            if terminated or truncated:
                s, info = env.reset()

        timestep_reward.append(total_reward)

        print(f"Episode: {episode}, steps: {t}, Total reward: {total_reward}")
    
    # plot(normalize(timestep_reward), epsilon_values) # normalized reward
    # plot(timestep_reward, epsilon_values)
    return timestep_reward, Q, epsilon_values

# ------------------------------------------------------------------------------------------------------------------------------
def SARSA(alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, n_tests):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = Q_value_initialize(n_states, n_actions, type = 0)
    timestep_reward = []
    epsilon_values = [] 
    epsilon_decay_rate = -np.log(EPS_END / EPS_START) / episodes
    
    for episode in range(episodes):
        print(f"Episode: {episode}")
        s, info = env.reset() # read also state
        
        epsilon_threshold = EPS_START * np.exp(-epsilon_decay_rate * episode)
        epsilon_values.append(epsilon_threshold)

        t = 0
        total_reward = 0
        while t < max_steps:
            t += 1
            a = epsilon_greedy(Q, epsilon_threshold, s)

            # Reward
            # -1 per step unless other reward is triggered.
            # +20 delivering passenger.
            # -10 executing “pickup” and “drop-off” actions illegally.
            s_, reward, terminated, truncated, info = env.step(a)

            #s_, reward, done, info = env.step(a)
            total_reward += reward
            a_next = epsilon_greedy(Q, epsilon, s_)

            if terminated or truncated:
                Q[s, a] += alpha * (reward - Q[s, a])
            else:
                Q[s, a] += alpha * (reward + (gamma * Q[s_, a_next]) - Q[s, a])
            s, a = s_, a_next
            
            if terminated or truncated:
                s, info = env.reset()

        timestep_reward.append(total_reward)

        print(f"Episode: {episode}, steps: {t}, Total reward: {total_reward}")
    
    # plot(normalize(timestep_reward), epsilon_values) # normalized reward
    # plot(timestep_reward, epsilon_values)
    return timestep_reward, Q, epsilon_values

# ----------------------------------------------------------------------------------------------------------------------
def ESARSA(alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, n_tests):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = Q_value_initialize(n_states, n_actions, type = 0)
    timestep_reward = []
    epsilon_values = [] 
    epsilon_decay_rate = -np.log(EPS_END / EPS_START) / episodes
    
    for episode in range(episodes):
        print(f"Episode: {episode}")
        s, info = env.reset() # read also state
        
        epsilon_threshold = EPS_START * np.exp(-epsilon_decay_rate * episode)
        epsilon_values.append(epsilon_threshold)

        t = 0
        total_reward = 0
        while t < max_steps:
            t += 1
            a = epsilon_greedy(Q, epsilon_threshold, s)

            # Reward
            # -1 per step unless other reward is triggered.
            # +20 delivering passenger.
            # -10 executing “pickup” and “drop-off” actions illegally.
            s_, reward, terminated, truncated, info = env.step(a)

            #s_, reward, done, info = env.step(a)
            total_reward += reward
            a_next = np.argmax(Q[s_, :]).item()

            if terminated or truncated:
                Q[s, a] += alpha * (reward - Q[s, a])
            else:
                Q[s, a] += alpha * (reward + (gamma * sum(Q[s_, a_next])) - Q[s, a])
            s, a = s_, a_next
            
            if terminated or truncated:
                s, info = env.reset()

        timestep_reward.append(total_reward)

        print(f"Episode: {episode}, steps: {t}, Total reward: {total_reward}")
    
    # plot(normalize(timestep_reward), epsilon_values) # normalized reward
    plot(timestep_reward, epsilon_values)
    return timestep_reward, Q
# -----------------------------------------------------------------------------------------
def test_agent(Q, n_tests = 1, delay=0.3, max_steps_test = 100):
    env = gym.make('Taxi-v3', render_mode="human") 
    #env = gym.make('CliffWalking-v0', render_mode="human")
    # env = gym.make('FrozenLake-v1', render_mode="human")
    # env = gym.make("Blackjack-v1", render_mode="human") # **Solve**

    for testing in range(n_tests):
        print(f"Test #{testing}")
        s, info = env.reset()
        t = 0
        while t < max_steps_test:
            t += 1
            time.sleep(delay)
            a = np.argmax(Q[s, :]).item()
            print(f"Chose action {a} for state {s}")
            s, reward, terminated, truncated, info = env.step(a)
            #time.sleep(1)

            if terminated or truncated:
                print("Finished!", reward)
                time.sleep(delay)
                break

if __name__ == "__main__":
    alpha = 0.3 # learning rate
    gamma = 0.95 # discount factor
    episodes = 800
    max_steps = 1500 

    epsilon = 0.01 # epsilon greedy exploration-explotation (higher more random)
    EPS_START = 1 
    EPS_END = 0.001
    Qlearning_timestep_reward, Q , Qlearning_epsilon_values = Qlearning(
        alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, n_tests=2)
    
    SARSA_timestep_reward, Q , SARSA_epsilon_values = SARSA(
        alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, n_tests=2)

    ESARSA_timestep_reward, Q , ESARSA_epsilon_values = ESARSA(
        alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, n_tests=2)
    
    plot(Qlearning_timestep_reward, Qlearning_epsilon_values, SARSA_timestep_reward, SARSA_epsilon_values)
  
    # Test policy (no learning)
    print(f"Q values:\n{Q}\nTesting now:")
    number_of_tests = 5
    # test_agent(Q, number_of_tests)