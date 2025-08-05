# Slide Example for Value Iteration (RL-Course NTNU, Saeedvand)
import numpy as np

states = 8 # number of states
A = ['lf', 'rf', 'lb', 'rb']  # actions
actions = 4

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

Value = [0, 0, 0, 0, 0, 0, 0, 0]  # Initial Value estimation of each state (we can set random too)
# Value = np.random.random(len(Value))                   # Generate random values for each state
# Value = np.random.randint(0, 10, size=len(Value))      # Random integers between 0 and 10
# Value = np.random.normal(0, 1, size=len(Value))         # Normal distribution with mean=0, std=1(standard deviation)
# print(Value)

#-------------------------------------
gamma = 0.95
delta = 0.01            # while values' changing are smaller than 'delta', stop the function.

for iteration in range(0, 100):     # 100 times of iteration
    NewValue = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(states):
        for a in range(actions):
            value_temp = 0          # The current time's value
            for j in range(states):
                value_temp += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value[j])    # Bellman's Equation for state-value
            NewValue[i] = round(max(NewValue[i], value_temp), 5)    
            # To compares 'NewValue' & 'value_temp', takes the maximum value to 5 decimal places.
    bellman_factor = 0
    for i in range(states):
        bellman_factor = max(bellman_factor, abs(Value[i]-NewValue[i]))
        # Maximum absolute value of "Value - NewValue"
    Value = NewValue        # Optimal Value
    print(iteration, NewValue, 'bellman_factor (' + str(bellman_factor) + ')' , sep=",      ")
    if(bellman_factor < delta):
        break

# Determine the policy (Only one iteration to return the maximum action sequence)
NewValue = [-1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1e10]
policy = ['NA','NA','NA','NA','NA', 'NA', 'NA', 'NA']
Terminal = ['','','','','', '', '', 'T']
for i in range(states):
    for a in range(actions):
        value_temp = 0
        for j in range(states):
            value_temp += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value[j])
        if(NewValue[i] < value_temp):
            if(Terminal[i] != 'T'):
                policy[i] = A[a]                                # The optimal policy
                NewValue[i] = max(NewValue[i], value_temp)
                # Determines witch action gives the highest value
            else:
                policy[i] = 'T'

print("The algoirthm's final policy is:", policy)