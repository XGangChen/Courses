# (A) : TD(0)?
#       TD(1)?  To compare
# (B) : What is the best value lambda
# (C) : Study alpha impact
# (D) : Change Texi-v3 to Desire
import numpy as np
import gymnasium as gym
import time
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')#, render_mode="human") #render_mode="human"
# env = gym.make('FrozenLake-v1', render_mode="human") #render_mode="human"
# env = gym.make('CliffWalking-v0', render_mode="human") #render_mode="human"

# lambda_ = 0.90 # trace-decay parameter, lambda weighting factor 
gamma_ = 0.7 # discount rate
alpha_ = 0.2
observation, info = env.reset() # initialize the environment

n_episodes = 10000
n_steps = 20

n_states, n_actions = env.observation_space.n, env.action_space.n

state_value = np.zeros(n_states)
policy = np.zeros(n_states, dtype=int)
e = np.zeros(n_states) # eligibility trace


def plot(td0_rewards, td1_rewards, td_lambda_rewards):
    plt.figure(2)
    plt.title('Average Value Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Average State-Value')
    plt.plot(td0_rewards, color='red', label='TD(0)')
    plt.plot(td1_rewards, color='blue', label='TD(1)')
    plt.plot(td_lambda_rewards, color='green', label='TD(Lambda)')
    plt.grid(axis='x', color='0.80')
    plt.legend(title='Parameter where:')
    plt.show()

td0_values = []
td1_values = []
td_lambda_values = []

num = 10

'''-----------------------------------------------------------TD(lambda)---------------------------------------------------------------'''
# Generate random initial policy
lambda_ = 0
sum_value = 0
best_lamda = 0
for n in range(num):
    
    lambda_ += 0.1
    for i in range(n_states):
        policy[i] = env.action_space.sample() # execute the policy

    for episode in range(n_episodes):
        observation, info = env.reset() # start with the environment's reset state
        action = policy[observation] # take first action from policy
        lambda_ = 0.90

        for t in range(n_steps):
            next_observation, reward, terminated, truncated, info = env.step(action) # based on policy generates new state
            next_action = policy[next_observation] # take next policy to execute

            if terminated or truncated:
                next_observation, info = env.reset()

            # Based on eligibility compute the TD-error and update every state's value estimate
            delta = reward + gamma_ * state_value[next_observation] - state_value[observation]

            # Update the eligibility of observed state
            e[observation] += 1.0

            # Update all the eligibilities using numpy vectorized operation
            state_value = state_value + alpha_ * delta * e


            # Decay the eligibility trace
            e = gamma_ * lambda_ * e

            # Move to the next state
            observation = next_observation
            action = next_action

        # Append the average state value for plotting
        td_lambda_values.append(np.mean(state_value))

        # print(episode, td_lambda_values[episode])
    if sum_value >= sum(td_lambda_values):
        best_lamda = lambda_

print(best_lamda)


# '''-------------------------------------------------------------TD(0)---------------------------------------------------------'''

# observation, info = env.reset() # initialize the environment

# n_states, n_actions = env.observation_space.n, env.action_space.n

# state_value = np.zeros(n_states)
# policy = np.zeros(n_states, dtype=int)
# e = np.zeros(n_states) # eligibility trace

# for i in range(n_states):
#     policy[i] = env.action_space.sample() # execute the policy

# for episode in range(n_episodes):
#     observation, info = env.reset() # start with the environment's reset state
#     action = policy[observation] # take first action from policy
#     lambda_ = 0

#     for t in range(n_steps):
#         next_observation, reward, terminated, truncated, info = env.step(action) # based on policy generates new state
#         next_action = policy[next_observation] # take next policy to execute

#         if terminated or truncated:
#             next_observation, info = env.reset()

#         # Based on eligibility compute the TD-error and update every state's value estimate
#         delta = reward + gamma_ * state_value[next_observation] - state_value[observation]

#         # Update the eligibility of observed state
#         e[observation] += 1.0

#         # Update all the eligibilities using numpy vectorized operation
#         state_value = state_value + alpha_ * delta * e


#         # Decay the eligibility trace
#         e = gamma_ * lambda_ * e

#         # Move to the next state
#         observation = next_observation
#         action = next_action

#     # Append the average state value for plotting
#     td0_values.append(np.mean(state_value))
#     # print(episode, td0_values[episode])

# '''----------------------------------------------------------------TD(1)--------------------------------------------------------------------'''
# observation, info = env.reset() # initialize the environment

# n_states, n_actions = env.observation_space.n, env.action_space.n

# state_value = np.zeros(n_states)
# policy = np.zeros(n_states, dtype=int)
# e = np.zeros(n_states) # eligibility trace

# for i in range(n_states):
#     policy[i] = env.action_space.sample() # execute the policy

# for episode in range(n_episodes):
#     observation, info = env.reset() # start with the environment's reset state
#     action = policy[observation] # take first action from policy
#     lambda_ = 1

#     for t in range(n_steps):
#         next_observation, reward, terminated, truncated, info = env.step(action) # based on policy generates new state
#         next_action = policy[next_observation] # take next policy to execute

#         if terminated or truncated:
#             next_observation, info = env.reset()

#         # Based on eligibility compute the TD-error and update every state's value estimate
#         delta = reward + gamma_ * state_value[next_observation] - state_value[observation]

#         # Update the eligibility of observed state
#         e[observation] += 1.0

#         # Update all the eligibilities using numpy vectorized operation
#         state_value = state_value + alpha_ * delta * e


#         # Decay the eligibility trace
#         e = gamma_ * lambda_ * e

#         # Move to the next state
#         observation = next_observation
#         action = next_action

#     # Append the average state value for plotting
#     td1_values.append(np.mean(state_value))
#     # print(episode, td1_values[episode])





plot(td0_values, td1_values, td_lambda_values)                                                    
 
env.close()