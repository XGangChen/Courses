import gymnasium as gym
import numpy as np
import time
import math
import matplotlib.pyplot as plt


env = gym.make('Taxi-v3')#, render_mode="human") 

def plot(rewards, epsilon_decay):
    fig, ax1 = plt.subplots()

    color = 'tab:green'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color=color)
    ax1.plot(rewards, color=color, label='Reward')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Epsilon', color=color)  # we already handled the x-label with ax1
    ax2.plot(epsilon_decay, color=color, label='Epsilon Decay')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Total Reward and Epsilon Decay - Q-Learning')
    plt.grid(axis='x', color='0.80')
    plt.show()

# This function initializes the Q-table (a 2D array) to store values for each state-action pair.
def Q_value_initialize(state, action, type = 0):
    if type == 1:
        return np.ones((state, action))
    elif type == 0:
        return np.zeros((state, action))
    elif type == -1:
        return np.random.random((state, action))
   
# This implements the epsilon-greedy policy.
def epsilon_greedy(Q, epsilon, s):
    if np.random.rand() > epsilon:
        action = np.argmax(Q[s, :]).item() # Exploits (chooses the action with the highest Q-value)
    else:
        action = env.action_space.sample() # Explores (chooses a random action) with probability epsilon

    return action

# To scale the rewards to a range of [0, 1].
def normalize(list): # you can use this to normalize your plot values
    xmin = min(list) 
    xmax = max(list)
    for i, x in enumerate(list):
        list[i] = (x-xmin) / (xmax-xmin)
    return list 

# Initializes the Q-table and sets up the environment.
def Qlearning(alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, n_tests):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = Q_value_initialize(n_states, n_actions, type = 0) # Q-table
    timestep_reward = []
    epsilon_values = [] 
    epsilon_decay_rate = -np.log(EPS_END / EPS_START) / episodes
    
    for episode in range(episodes):
        print(f"Episode: {episode}")
        # The environment is reset, and an initial state s is retrieved.
        s, info = env.reset() # read also state
        # The agent selects actions according to the epsilon-greedy policy.
        epsilon_threshold = EPS_START * np.exp(-epsilon_decay_rate * episode)
        epsilon_values.append(epsilon_threshold)

        t = 0
        total_reward = 0
        while t < max_steps: # The process repeats until the episode terminates or the max steps are reached.
            t += 1
            a = epsilon_greedy(Q, epsilon_threshold, s)

            # Reward
            # -1 per step unless other reward is triggered.
            # +20 delivering passenger.
            # -10 executing “pickup” and “drop-off” actions illegally.
            s_, reward, terminated, truncated, info = env.step(a)
            # The environment responds with the next state (s_), reward, and whether the episode has terminated (done).

            #s_, reward, done, info = env.step(a)
            total_reward += reward
            a_next = np.argmax(Q[s_, :]).item()

            if terminated or truncated:
                Q[s, a] += alpha * (reward - Q[s, a])
            else:
                Q[s, a] += alpha * (reward + (gamma * Q[s_, a_next]) - Q[s, a])
            s, a = s_, a_next
            
            if terminated or truncated:
                s, info = env.reset()

        # Rewards for each episode are collected to track learning progress.
        timestep_reward.append(total_reward)

        print(f"Episode: {episode}, steps: {t}, Total reward: {total_reward}")
    
    # plot(normalize(timestep_reward), epsilon_values) # normalized reward
    plot(timestep_reward, epsilon_values)
    return timestep_reward, Q

#----------------------------------------------------
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
            a = np.argmax(Q[s, :]).item() # The Q-table is used to choose actions 
            print(f"Chose action {a} for state {s}")
            s, reward, terminated, truncated, info = env.step(a)
            #time.sleep(1)

            if terminated or truncated:
                print("Finished!", reward)
                time.sleep(delay)
                break

# '''--------------------------------Original Main Code----------------------------------------------
if __name__ == "__main__":
    alpha = 0.3 # learning rate
    gamma = 0.95 # discount factor
    episodes = 800 # The number of episodes (iterations) over which the algorithm runs.
    max_steps = 1500 # Limits the number of steps per episode.

    epsilon = 0.01 # epsilon greedy exploration-explotation (higher more random)

    # These control the starting and ending values for epsilon decay.
    EPS_START = 1 
    EPS_END = 0.001
    
    timestep_reward, Q = Qlearning(
        alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, n_tests=2)
  
    # Test policy (no learning)
    print(f"Q values:\n{Q}\nTesting now:")
    number_of_tests = 5
    test_agent(Q, number_of_tests)
# '''

'''---------------------------------------Q3_A-----------------------------------------------------
def Qlearning_variable_alpha(alpha_start, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, n_tests):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = Q_value_initialize(n_states, n_actions, type=0)  # Q-table
    timestep_reward = []
    epsilon_values = [] 
    epsilon_decay_rate = -np.log(EPS_END / EPS_START) / episodes

    # Parameters for variable alpha
    k = 0.001  # Decay rate for learning rate

    for episode in range(episodes):
        print(f"Episode: {episode}")
        s, info = env.reset()  # Reset environment and retrieve initial state

        # Calculate variable learning rate for this episode
        alpha = alpha_start / (1 + k * episode)

        # Epsilon decay
        epsilon_threshold = EPS_START * np.exp(-epsilon_decay_rate * episode)
        epsilon_values.append(epsilon_threshold)

        t = 0
        total_reward = 0
        while t < max_steps:
            t += 1
            a = epsilon_greedy(Q, epsilon_threshold, s)

            s_, reward, terminated, truncated, info = env.step(a)
            total_reward += reward
            a_next = np.argmax(Q[s_, :]).item()

            if terminated or truncated:
                Q[s, a] += alpha * (reward - Q[s, a])
            else:
                Q[s, a] += alpha * (reward + (gamma * Q[s_, a_next]) - Q[s, a])

            s = s_

            if terminated or truncated:
                break

        timestep_reward.append(total_reward)
        print(f"Episode: {episode}, steps: {t}, Total reward: {total_reward}")

    plot(timestep_reward, epsilon_values)
    return timestep_reward, Q

if __name__ == "__main__":
    alpha_start = 0.3  # Initial learning rate
    gamma = 0.95  # Discount factor
    episodes = 800  # Number of episodes
    max_steps = 1500  # Max steps per episode

    epsilon = 0.01  # Exploration-exploitation trade-off
    EPS_START = 1
    EPS_END = 0.001

    timestep_reward, Q = Qlearning_variable_alpha(
        alpha_start, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, n_tests=2)

    # Test policy (no learning)
    print(f"Q values:\n{Q}\nTesting now:")
    number_of_tests = 5
    test_agent(Q, number_of_tests)
'''

'''----------------------------------------------Q3_B------------------------------------------------------
def Qlearning_variable_alpha(alpha_start, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, n_tests):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = Q_value_initialize(n_states, n_actions, type=0)  # Q-table
    timestep_reward = []
    epsilon_values = [] 
    epsilon_decay_rate = -np.log(EPS_END / EPS_START) / episodes

    # Parameters for variable alpha
    k = 0.001  # Decay rate for learning rate

    for episode in range(episodes):
        print(f"Episode: {episode}")
        s, info = env.reset()  # Reset environment and retrieve initial state

        # Calculate variable learning rate for this episode
        alpha = alpha_start / (1 + k * episode)

        # Epsilon decay
        epsilon_threshold = EPS_START * np.exp(-epsilon_decay_rate * episode)
        epsilon_values.append(epsilon_threshold)

        t = 0
        total_reward = 0
        while t < max_steps:
            t += 1
            a = epsilon_greedy(Q, epsilon_threshold, s)

            s_, reward, terminated, truncated, info = env.step(a)

            # Novel reward dynamics
            if reward == 20:  # Successfully delivered passenger
                reward += 10  # Bonus reward for successful delivery
            elif reward == -10:  # Illegal pickup/drop-off action
                reward -= 5  # Extra penalty for illegal actions
            elif t >= max_steps // 2 and not terminated:  # Encourage faster completion
                reward -= 1  # Small penalty for being slow

            total_reward += reward
            a_next = np.argmax(Q[s_, :]).item()

            if terminated or truncated:
                Q[s, a] += alpha * (reward - Q[s, a])
            else:
                Q[s, a] += alpha * (reward + (gamma * Q[s_, a_next]) - Q[s, a])

            s = s_

            if terminated or truncated:
                break

        timestep_reward.append(total_reward)
        print(f"Episode: {episode}, steps: {t}, Total reward: {total_reward}")

    plot(timestep_reward, epsilon_values)
    return timestep_reward, Q

if __name__ == "__main__":
    alpha_start = 0.3  # Initial learning rate
    gamma = 0.95  # Discount factor
    episodes = 800  # Number of episodes
    max_steps = 1500  # Max steps per episode

    epsilon = 0.01  # Exploration-exploitation trade-off
    EPS_START = 1
    EPS_END = 0.001

    timestep_reward, Q = Qlearning_variable_alpha(
        alpha_start, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, n_tests=2)

    # Test policy (no learning)
    print(f"Q values:\n{Q}\nTesting now:")
    number_of_tests = 5
    test_agent(Q, number_of_tests)
'''

'''------------------------------------------------Q3_C---------------------------------------------
# Define a new exploration strategy: Upper Confidence Bound (UCB)
def ucb_exploration(Q, state, N, c=2):
    """
    Implements UCB exploration strategy.
    Q: Q-table (state-action values)
    state: current state
    N: state-action visit counts (2D array)
    c: exploration constant, controls exploration vs. exploitation
    """
    action_counts = N[state, :]
    total_counts = np.sum(action_counts) + 1  # Avoid divide by zero

    ucb_values = Q[state, :] + c * np.sqrt(np.log(total_counts) / (action_counts + 1))
    return np.argmax(ucb_values)

def Qlearning_variable_alpha(alpha_start, gamma, episodes, max_steps, EPS_START, EPS_END, n_tests):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = Q_value_initialize(n_states, n_actions, type=0)  # Q-table
    N = Q_value_initialize(n_states, n_actions, type=0)  # Visit counts for UCB
    timestep_reward = []

    # Parameters for variable alpha
    k = 0.001  # Decay rate for learning rate

    for episode in range(episodes):
        print(f"Episode: {episode}")
        s, info = env.reset()  # Reset environment and retrieve initial state

        # Calculate variable learning rate for this episode
        alpha = alpha_start / (1 + k * episode)

        t = 0
        total_reward = 0
        while t < max_steps:
            t += 1
            a = ucb_exploration(Q, s, N)  # Use UCB exploration

            s_, reward, terminated, truncated, info = env.step(a)

            # Novel reward dynamics
            if reward == 20:  # Successfully delivered passenger
                reward += 10  # Bonus reward for successful delivery
            elif reward == -10:  # Illegal pickup/drop-off action
                reward -= 5  # Extra penalty for illegal actions
            elif t >= max_steps // 2 and not terminated:  # Encourage faster completion
                reward -= 1  # Small penalty for being slow

            total_reward += reward
            a_next = np.argmax(Q[s_, :]).item()

            if terminated or truncated:
                Q[s, a] += alpha * (reward - Q[s, a])
            else:
                Q[s, a] += alpha * (reward + (gamma * Q[s_, a_next]) - Q[s, a])

            # Update visit counts for UCB
            N[s, a] += 1
            s = s_

            if terminated or truncated:
                break

        timestep_reward.append(total_reward)
        print(f"Episode: {episode}, steps: {t}, Total reward: {total_reward}")

    plot(timestep_reward, [EPS_START] * episodes)  # Plot rewards (UCB doesn't decay epsilon)
    return timestep_reward, Q

if __name__ == "__main__":
    alpha_start = 0.3  # Initial learning rate
    gamma = 0.95  # Discount factor
    episodes = 800  # Number of episodes
    max_steps = 1500  # Max steps per episode

    timestep_reward, Q = Qlearning_variable_alpha(
        alpha_start, gamma, episodes, max_steps, EPS_START=1, EPS_END=0.001, n_tests=2)

    # Test policy (no learning)
    print(f"Q values:\n{Q}\nTesting now:")
    number_of_tests = 5
    test_agent(Q, number_of_tests)
'''

'''------------------------------------------Q3_D---------------------------------------------------------
# Define a new exploration strategy: Upper Confidence Bound (UCB)
def ucb_exploration(Q, state, N, c=2):
    """
    Implements UCB exploration strategy.
    Q: Q-table (state-action values)
    state: current state
    N: state-action visit counts (2D array)
    c: exploration constant, controls exploration vs. exploitation
    """
    action_counts = N[state, :]
    total_counts = np.sum(action_counts) + 1  # Avoid divide by zero

    ucb_values = Q[state, :] + c * np.sqrt(np.log(total_counts) / (action_counts + 1))
    return np.argmax(ucb_values)

def Qlearning_variable_alpha(alpha_start, gamma, episodes, max_steps, EPS_START, EPS_END, n_tests):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = Q_value_initialize(n_states, n_actions, type=0)  # Q-table
    N = Q_value_initialize(n_states, n_actions, type=0)  # Visit counts for UCB
    timestep_reward = []

    # Parameters for variable alpha
    k = 0.001  # Decay rate for learning rate

    for episode in range(episodes):
        print(f"Episode: {episode}")
        s, info = env.reset()  # Reset environment and retrieve initial state

        # Calculate variable learning rate for this episode
        alpha = alpha_start / (1 + k * episode)

        t = 0
        total_reward = 0
        while t < max_steps:
            t += 1
            a = ucb_exploration(Q, s, N)  # Use UCB exploration

            s_, reward, terminated, truncated, info = env.step(a)

            # Novel reward dynamics
            if reward == 20:  # Successfully delivered passenger
                reward += 10  # Bonus reward for successful delivery
            elif reward == -10:  # Illegal pickup/drop-off action
                reward -= 5  # Extra penalty for illegal actions
            elif t >= max_steps // 2 and not terminated:  # Encourage faster completion
                reward -= 1  # Small penalty for being slow

            total_reward += reward
            a_next = np.argmax(Q[s_, :]).item()

            if terminated or truncated:
                Q[s, a] += alpha * (reward - Q[s, a])
            else:
                Q[s, a] += alpha * (reward + (gamma * Q[s_, a_next]) - Q[s, a])

            # Update visit counts for UCB
            N[s, a] += 1
            s = s_

            if terminated or truncated:
                break

        timestep_reward.append(total_reward)
        print(f"Episode: {episode}, steps: {t}, Total reward: {total_reward}")

    plot(timestep_reward, [EPS_START] * episodes)  # Plot rewards (UCB doesn't decay epsilon)
    return timestep_reward, Q

if __name__ == "__main__":
    alpha_start = 0.3  # Initial learning rate
    gamma = 0.95  # Discount factor
    episodes = 800  # Number of episodes
    max_steps = 1500  # Max steps per episode

    timestep_reward, Q = Qlearning_variable_alpha(
        alpha_start, gamma, episodes, max_steps, EPS_START=1, EPS_END=0.001, n_tests=2)

    # Test policy (no learning)
    print(f"Q values:\n{Q}\nTesting now:")
    number_of_tests = 5
    test_agent(Q, number_of_tests)
'''
