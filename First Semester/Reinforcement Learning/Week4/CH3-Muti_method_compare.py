import numpy as np
import matplotlib.pyplot as plt

states = 8  # number of states
actions = 4
A = ['lf', 'rf', 'lb', 'rb']  # actions
Terminal = ['','','','','', '', '', 'T']

gamma = 0.95
delta = 0.01  # Convergence threshold

# Rewards
Reward = [[0, 0, -1, -1], [2, 2, -1, -1], [1, 1, -1, -1], [-1, -1, -1, -1], [3, 3, -1, -1], [-3, -3, -1, -1], [-7, -7, -1, -1], [5, 5, -1, -1]]

# Transition Probabilities
TransitionProbability = [
    [[0.0, 0.0, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0], [0.3, 0.7, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    [[0.2, 0.1, 0.0, 0.7], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.7, 0.2, 0.0, 0.1], [0.2, 0.7, 0.0, 0.1], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    [[0.2, 0.1, 0.7, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.7, 0.2, 0.1, 0.0], [0.2, 0.7, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0, 0.0], [0.2, 0.1, 0.6, 0.1], [0.2, 0.1, 0.1, 0.6], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.6, 0.2, 0.1, 0.1], [0.2, 0.6, 0.1, 0.1]],
    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.2, 0.1, 0.7, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.2, 0.0]],
    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
]

def value_iteration():
    Value = [0] * states
    iterations = 0
    value_history = []

    while True:
        NewValue = [0] * states
        for i in range(states):
            for a in range(actions):
                value_temp = 0
                for j in range(states):
                    value_temp += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value[j])
                NewValue[i] = max(NewValue[i], value_temp)

        bellman_factor = max(abs(Value[i] - NewValue[i]) for i in range(states))
        Value = NewValue
        iterations += 1
        value_history.append(sum(Value))  # Track total value for plotting

        if bellman_factor < delta:
            break

        # Determine the policy (Only one iteration to return the maximum action sequence)
    NewValue = [-1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1e10]
    VI_policy = ['NA','NA','NA','NA','NA', 'NA', 'NA', 'NA']
    for i in range(states):
        for a in range(actions):
            value_temp = 0
            for j in range(states):
                value_temp += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value[j])
            if(NewValue[i] < value_temp):
                VI_policy[i] = A[a]                                # The optimal policy
                NewValue[i] = max(NewValue[i], value_temp)
                # Determines witch action gives the highest value
    return iterations, value_history, VI_policy

def policy_iteration():
    PI_policy = ['lf']*states
    # Terminal = ['','','','','', '', '', 'T']
    Value = [0] * states
    iterations = 0
    value_history = []

    while True:
        # Policy Evaluation
        while True:
            max_change = 0
            for i in range(states):
                old_value = Value[i]
                action = A.index(PI_policy[i])
                value_temp = 0
                for j in range(states):
                    value_temp += TransitionProbability[i][j][action] * (Reward[j][action] + gamma * Value[j])
                Value[i] = value_temp
                max_change = max(max_change, abs(old_value - value_temp))
            if max_change < delta:
                break

        # Policy Improvement
        policy_stable = True
        for i in range(states):
            old_action = PI_policy[i]
            best_value = -1e10
            for a in range(actions):
                value_temp = 0
                for j in range(states):
                    value_temp += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value[j])
                if value_temp > best_value:
                    best_action = A[a]
                    # if (Terminal[i] != 'T'):
                    # else:
                    #     best_action = 'T'
                    best_value = value_temp
            PI_policy[i] = best_action
            if old_action != best_action:
                policy_stable = False
        iterations += 1
        value_history.append(sum(Value))  # Track total value for plotting
        if policy_stable:
            break
    return iterations, value_history, PI_policy

def modified_policy_iteration(evaluation_steps=3):
    MPI_policy = ['lf'] * states
    Value = [0] * states
    iterations = 0
    value_history = []

    while True:
        # Partial Policy Evaluation
        for _ in range(evaluation_steps):
            for i in range(states):
                action = A .index(MPI_policy[i])
                value_temp = 0
                for j in range(states):
                    value_temp += TransitionProbability[i][j][action] * (Reward[j][action] + gamma * Value[j])
                Value[i] = value_temp

        # Policy Improvement
        policy_stable = True
        for i in range(states):
            old_action = MPI_policy[i]
            best_value = -1e10
            for a in range(actions):
                value_temp = 0
                for j in range(states):
                    value_temp += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value[j])
                if value_temp > best_value:
                    best_value = value_temp
                    best_action = A[a]
            MPI_policy[i] = best_action
            if old_action != best_action:
                policy_stable = False

        iterations += 1
        value_history.append(sum(Value))  # Track total value for plotting
        if policy_stable:
            break

    return iterations, value_history, MPI_policy

# Run all three algorithms
vi_iterations, vi_history, vi_policy = value_iteration()
pi_iterations, pi_history, pi_policy = policy_iteration()
mpi_iterations, mpi_history, mpi_policy = modified_policy_iteration()

for i in range(states):
    if(Terminal[i] == 'T'):
        vi_policy[i] = 'T'
        pi_policy[i] = 'T'
        mpi_policy[i] = 'T'

print("The algoirthm's final policy of Value Iteration is:", vi_policy)
print("The algoirthm's final policy of Policy Iteration is:", pi_policy)
print("The algoirthm's final policy of Modified Policy Iteration is:", mpi_policy)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(range(vi_iterations), vi_history, label='Value Iteration', marker='o')
plt.plot(range(pi_iterations), pi_history, label='Policy Iteration', marker='s')
plt.plot(range(mpi_iterations), mpi_history, label='Modified Policy Iteration', marker='^')
plt.xlabel('Iterations')
plt.ylabel('Sum of State Values')
plt.title('Convergence Comparison of Value Iteration, Policy Iteration, and Modified Policy Iteration')
plt.legend()
plt.grid()
plt.show()