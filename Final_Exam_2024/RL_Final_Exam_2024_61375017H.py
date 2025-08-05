# Slide Example for Q-Learning (RL-Course NTNU, Saeedvand)

import gymnasium as gym
import numpy as np
import time
import math
import matplotlib.pyplot as plt


# env = gym.make('Taxi-v3')#, render_mode="human") 
env = gym.make('FrozenLake-v1')
# env = gym.make('CliffWalking-v0')#, render_mode="human") 
# env = gym.make('Myslide-v1')  # Important: The player may slip (20% error).
# env = gym.make("Blackjack-v1") # **Practice, Solve**


def plot(rewards, epsilon_decay, td_value):
    fig, ax1 = plt.subplots()

    color = 'tab:green'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color=color)
    ax1.plot(rewards, color=color, label='Reward')
    ax1.plot(td_value, color='red', label='TD Value')
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
def Qlearning(lambda_, alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, n_tests):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = Q_value_initialize(n_states, n_actions, type = 0) # Q-table
    timestep_reward = []
    epsilon_values = [] 
    epsilon_decay_rate = -np.log(EPS_END / EPS_START) / episodes

    e = np.zeros(n_states) # eligibility trace
    state_value = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)
    td_lambda_values = []

    sum_value = 0
    best_lamda = 0

    lambda_ += 0.1
    for i in range(n_states):
        policy[i] = env.action_space.sample() # execute the policy
    
    for episode in range(episodes):
        print(f"Episode: {episode}")
        # The environment is reset, and an initial state s is retrieved.
        s, info = env.reset() # read also state
        s_lambda, info = env.reset() # read also state
        # The agent selects actions according to the epsilon-greedy policy.
        epsilon_threshold = EPS_START * np.exp(-epsilon_decay_rate * episode)
        epsilon_values.append(epsilon_threshold)

        a_lambda = policy[s_lambda] # take first action from policy
        lambda_ = 0.90

        t = 0
        total_reward = 0
        total_reward_lambda = 0
        while t < max_steps: # The process repeats until the episode terminates or the max steps are reached.
            t += 1
            a = epsilon_greedy(Q, epsilon_threshold, s)
            

            # Reward
            # -1 per step unless other reward is triggered.
            # +20 delivering passenger.
            # -10 executing “pickup” and “drop-off” actions illegally.
            s_, reward, terminated, truncated, info = env.step(a)
            s_lambda_next, reward_lambda, terminated, truncated, info = env.step(a_lambda)
            # The environment responds with the next state (s_), reward, and whether the episode has terminated (done).
            a_next = np.argmax(Q[s_, :]).item()
            a_next_lambda = policy[s_lambda_next] # take next policy to execute
            # a_next = policy[s_] # take next policy to execute

            # Based on eligibility compute the TD-error and update every state's value estimate
            delta = reward_lambda + gamma * state_value[s_lambda_next] - state_value[s_lambda]

            # Update the eligibility of observed state
            e[s_lambda] += 1.0
            #s_, reward, done, info = env.step(a)
            total_reward += reward
            total_reward_lambda += reward_lambda


            # Update all the eligibilities using numpy vectorized operation
            state_value = state_value + alpha * delta * e

            # Decay the eligibility trace
            e = gamma * lambda_ * e

            if terminated or truncated:
                Q[s, a] += alpha * (reward - Q[s, a])
            else:
                Q[s, a] += alpha * (reward + (gamma * Q[s_, a_next]) - Q[s, a])
            s, a, s_lambda, a_lambda = s_, a_next, s_lambda_next, a_next_lambda
            
            if terminated or truncated:
                s, info = env.reset()
                s_lambda, info = env.reset()

        # Append the average state value for plotting
        td_lambda_values.append(np.mean(state_value))
        # Rewards for each episode are collected to track learning progress.
        timestep_reward.append(total_reward)
        print(episode, td_lambda_values[episode])

        print(f"Episode: {episode}, steps: {t}, Total reward: {total_reward}")
    
    if sum_value >= sum(td_lambda_values):
        best_lamda = lambda_
    # plot(normalize(timestep_reward), epsilon_values) # normalized reward
    plot(timestep_reward, epsilon_values, td_lambda_values)
    return timestep_reward, Q, best_lamda

#----------------------------------------------------
def test_agent(Q, n_tests = 1, delay=0.3, max_steps_test = 100):
    # env = gym.make('Taxi-v3', render_mode="human") 
    #env = gym.make('CliffWalking-v0', render_mode="human")
    env = gym.make('FrozenLake-v1', render_mode="human")
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

if __name__ == "__main__":
    alpha = 0.3 # learning rate
    gamma = 0.95 # discount factor
    episodes = 800 # The number of episodes (iterations) over which the algorithm runs.
    max_steps = 1500 # Limits the number of steps per episode.

    epsilon = 0.01 # epsilon greedy exploration-explotation (higher more random)

    lambda_ = 0

    # These control the starting and ending values for epsilon decay.
    EPS_START = 1 
    EPS_END = 0.001
    
    timestep_reward, Q, best_lamda = Qlearning(
        lambda_, alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, n_tests=2)
    
    print("best lambda is:", best_lamda)
    # Test policy (no learning)
    print(f"Q values:\n{Q}\nTesting now:")
    number_of_tests = 10
    test_agent(Q, number_of_tests)