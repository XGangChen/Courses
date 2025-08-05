# To compare "Value Iteration" & "Policy Iteration" & "Modified Policy Iteration"
import numpy as np
import matplotlib.pyplot as plt

states = 8 # number of states
actions = 4
A = ['lf', 'rf', 'lb', 'rb']  # actions
Terminal = ['','','','','', '', '', 'T']

gamma = 0.95
delta = 0.01            # while values' changing are smaller than 'delta', stop the function.

# In case that reward can be different to be in one state with different actions we can define them seperately, in our example both
# left and right actions lead to same reward in individual state [State, [State, action]] * in other examples can be different
#               S1             S2             S3               S4             S5               S6                 S7              S8
Reward = [[0, 0, -1, -1],[2, 2, -1, -1],[1, 1, -1, -1],[-1, -1, -1, -1],[3, 3, -1, -1], [-3, -3, -1, -1], [-7, -7, -1, -1], [5, 5, -1, -1]] 


TransitionProbability = [ #[Left_Front, Right_Front, Left_Back, Right_Back]
    #         S1                    S2                    S3                   S4                    S5                    S6                    S7                    S8
    [[0.0, 0.0, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0], [0.3, 0.7, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  
    # if you are in s1 the probability to go s2(LF=0.7, RF=0.3, LB=0.0, RB=0.0), and s3(LF=0.3, RF=0.7, LB=0.0, RB=0.0)
    [[0.2, 0.1, 0.0, 0.7], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.7, 0.2, 0.0, 0.1], [0.2, 0.7, 0.0, 0.1], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # s2 
    [[0.2, 0.1, 0.7, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.7, 0.2, 0.1, 0.0], [0.2, 0.7, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # s3 
    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # s4  back to last state
    [[0.0, 0.0, 0.0, 0.0], [0.2, 0.1, 0.6, 0.1], [0.2, 0.1, 0.1, 0.6], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.6, 0.2, 0.1, 0.1], [0.2, 0.6, 0.1, 0.1]],  # s5 
    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.2, 0.1, 0.7, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.2, 0.0]],  # s6  
    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # s7  back to last state
    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]   # s8  (terminal)
]

'''-------------------------------Value Iteration(Assesses the value directly, then get optimal policy)--------------------------------------'''
def value_iteration():
    Value = [0]*states
    value_history = []
    iteration = 0

    while True:
        NewValue = [0]*states
        bellman_factor = 0
        for i in range(states):
            for a in range(actions):
                value_temp = 0                                                                          # The current time's value
                for j in range(states):
                    value_temp += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value[j])    # Bellman's Equation for state-value
                NewValue[i] = max(NewValue[i], value_temp)                                    # To compares 'NewValue' & 'value_temp', takes the maximum value to 5 decimal places.
            bellman_factor = max(bellman_factor, abs(Value[i] - NewValue[i]))                                                 # Maximum absolute value of "Value - NewValue"
        Value = NewValue                                                                                # Optimal Value
        iteration += 1
        # print(iteration, NewValue, 'bellman_factor (' + str(bellman_factor) + ')', sep=",      ")
        value_history.append(sum(Value))
        if(bellman_factor < delta):
            break

    # Value Iteration (Only one iteration to return the maximum action sequence)
    NewValue = [-1e10]*states
    VIpolicy = ['NA']*states
    for i in range(states):
        for a in range(actions):
            value_temp = 0
            for j in range(states):
                value_temp += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value[j])
            if(NewValue[i] < value_temp):
                VIpolicy[i] = A[a]                              # The optimal policy
                NewValue[i] = max(NewValue[i], value_temp)      # Determines witch action gives the highest value
    return iteration, VIpolicy, value_history

'''----------------------------Policy Iteration (Iteratively evaluate and improve the policy until convergence)-------------------------------'''
def policy_iteration():
    PIpolicy = ['lf']*states
    Value = [0]*states
    value_history = []
    iteration = 0

    while True:
        # Policy Evaluation
        while True:
            bellman_factor = 0
            for i in range(states):
                old_value = Value[i]
                action = A.index(PIpolicy[i])
                value_temp = 0          # The current time's value
                for j in range(states):
                    value_temp += TransitionProbability[i][j][action] * (Reward[j][action] + gamma * Value[j])    # Bellman's Equation for state-value
                Value[i] = value_temp
                bellman_factor = max(bellman_factor, abs(old_value - value_temp))
            # print(iteration, NewValue, 'bellman_factor (' + str(bellman_factor) + ')', sep=",      ")
            if bellman_factor < delta:
                break
        
        # Policy Improvement
        policy_stable = True
        for i in range(states):
            old_action = PIpolicy[i]
            best_value = -1e10
            for a in range(actions):
                value_temp = 0
                for j in range(states):
                    value_temp += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value[j])
                if value_temp > best_value:
                    PIpolicy[i] = A[a]
                    best_value = value_temp
            if old_action != PIpolicy[i]:
                policy_stable = False
        print(Value)
        value_history.append(sum(Value))
        iteration += 1

        if policy_stable:
            break
    
    return iteration, PIpolicy, value_history


'''-----------------------Modified Policy Iteration (Repeat K times Policy evaluation step, then Policy improvement step)----------------------'''
def modified_poliocy_iteration(evaluation_steps = 3):
    MPIpolicy = ['lf']*states
    Value = [0]*states
    valuee_history = []
    iteration = 0

    while True:
        # Partial Policy Evaluation
        for _ in range(evaluation_steps):
            for i in range(states):
                action = A.index(MPIpolicy[i])
                value_temp = 0
                for j in range(states):
                    value_temp += TransitionProbability[i][j][action] * (Reward[j][action] + gamma * Value[j])
                Value[i] = value_temp
        
        # Policy Improvement
        policy_stable = True
        for i in range(states):
            old_action = MPIpolicy[i]
            best_value = -1e10
            for a in range(actions):
                value_temp = 0
                for j in range(states):
                    value_temp += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value[j])
                if value_temp > best_value:
                    best_value = value_temp
                    MPIpolicy[i] = A[a]
            
            if old_action != MPIpolicy[i]:
                policy_stable = False
        print(Value)
        valuee_history.append(sum(Value))
        iteration += 1

        if policy_stable:
            break
    return iteration, MPIpolicy, valuee_history


vi_iteration, vi_policy, vi_history = value_iteration()
pi_iteration, pi_policy, pi_history = policy_iteration()
mpi_iteration, mpi_policy, mpi_history = modified_poliocy_iteration()

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
plt.plot(range(vi_iteration), vi_history, label='Value Iteration', marker='o')
plt.plot(range(pi_iteration), pi_history, label='Policy Iteration', marker='s')
plt.plot(range(mpi_iteration), mpi_history, label='Modified Policy Iteration', marker='^')
plt.xlabel('Iterations')
plt.ylabel('Sum of State Values')
plt.title('Convergence Comparison of Value Iteration, Policy Iteration, and Modified Policy Iteration')
plt.legend()
plt.grid()
plt.show()