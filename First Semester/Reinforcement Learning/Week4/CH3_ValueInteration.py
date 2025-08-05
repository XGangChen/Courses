# Slide Example for Value Iteration (RL-Course NTNU, Saeedvand)

states = 8 # number of states
A = ['l', 'r']  # actions
actions = 2

# In case that reward can be different to be in one state with different actions we candefine them seperately, in our example both
# left and rith actions lead to same reward in individual state [State, [State, action]] * in other examples can be differentCH3_ValueInteration.py
#           S1     S2     S3       S4     S5        S6       S7       S8
Reward = [[0, 0],[2, 2],[1, 1],[-1, -1],[3, 3], [-3, -3], [-7, -7], [5, 5]] 


TransitionProbability = [ #[Left, Right]
    #    S1          S2          S3          S4          S5          S6          S7          S8
    [[0.0, 0.0], [0.7, 0.3], [0.3, 0.7], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # if you are in s1 the probability to go s2(L=0.7, R=0.3), and s3(L=0.3, R=0.7)
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.7, 0.3], [0.3, 0.7], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # s2 
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.7, 0.3], [0.3, 0.7], [0.0, 0.0], [0.0, 0.0]],  # s3 
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # s4  (terminal) all zero
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.7, 0.3], [0.3, 0.7]],  # s5 
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]],  # s6  100% to left
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # s7  (terminal)
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]   # s8  (terminal)
]

Value = [0, 0, 0, 0, 0, 0, 0, 0]  # Initial Value estimation of each state (we can set random too)

#-------------------------------------

gamma = 0.9
bellman_factor = 0
delta = 0.001
NewValue = [-1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1e10]

for iteration in range(0, 100):
    NewValue = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(states):
        for a in range(actions):
            value_temp = 0
            for j in range(states):
                value_temp += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value[j])
            NewValue[i] = round(max(NewValue[i], value_temp), 5)
    bellman_factor = 0
    for i in range(states):
        bellman_factor = max(bellman_factor, abs(Value[i]-NewValue[i]))
    Value = NewValue
    print(iteration, NewValue, 'bellman_factor (' + str(bellman_factor) + ')' , sep=",    ")
    if(bellman_factor < delta):
        break

# Determine the policy (One time iteration)
NewValue = [-1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1e10]
policy = ['NA','NA','NA','NA','NA', 'NA', 'NA', 'NA']
Terminal = ['','','','T','', '', 'T', 'T']
for i in range(states):
    for a in range(actions):
        value_temp = 0
        for j in range(states):
            value_temp += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value[j])
        if(NewValue[i] < value_temp):
            if(Terminal[i] != 'T'):
                policy[i] = A[a]
                NewValue[i] = max(NewValue[i], value_temp)
            else:
                policy[i] = 'T'

print("The algoirthm's final policy is:", policy)