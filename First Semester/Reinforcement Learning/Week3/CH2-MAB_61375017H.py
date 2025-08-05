import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Data visualization library based on matplotlib
import scipy.stats as stats     
#This module contains a large number of probability distributions, summary and frequency statistics, correlation functions and statistical tests, 
#masked statistics, kernel density estimation, quasi-Monte Carlo functionality, and more.



np.random.seed(20) # Numerical value that generates a new set or repeats pseudo-random numbers. 
#The value in the numpy random seed saves the state of randomness.

# The probability of winning (exact value for each bandit), you can add more bandits here
Number_of_Bandits = 4
p_bandits = [0.5, 0.1, 0.8, 0.3] # Probability of Wining for each bandit. 
    #Color: Blue, Orange, Green, Red #Note: I gave big values to only visualize better, in real machine chance is very slim

def bandit_run(index):
    if np.random.rand() >= p_bandits[index]: #random  probability to win or lose per machine
        return 0 # Lose
    else:
        return 1 # Win

def Thompson_plot(distribution, step, ax):
    plt.figure(1)
    plot_x = np.linspace(0.000, 1, 200) # create sequences of evenly spaced numbers structured as a NumPy array. 
    # numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
    #np.linspace(start, stop, how many number within 0.000 to 1)
    ax.set_title(f'Step {step:d}')
    for d in distribution:
        y = d.pdf(plot_x)
        ax.plot(plot_x, y) # draw the curve of the plot
        ax.fill_between(plot_x, y, 0, alpha=0.1) # fill under the curve of the plot, 
                                                    #"0"=>The baseline y-value where the filling will start, "alpha=>transparency level"
    ax.set_ylim(bottom = 0) # limit plot axis

figure, ax = plt.subplots(4, 3, figsize=(9, 7)) # set the number of the plots in row and column and their sizes
ax = ax.flat # Iterator to plot

def plot_rewards(Thompson_rewards, UCB_rewards, e_greedy_rewards):
    plt.figure(2)
    plt.title('Rewards Comparision')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.plot(Thompson_rewards, color='green', label='Thompson')
    plt.plot(UCB_rewards, color='blue', label='UCB')
    plt.plot(e_greedy_rewards, color='red', label='e_greedy')
    plt.grid(axis='x', color='0.80')        #To display grid lines on the plot
    plt.legend(title='Parameter where:')
    plt.show()

N = 1000        # number of steps
epsilon = 0.1   #e for the e-greedy algorithm

#   Initialize for Thompson Sampling
Thompson_runing_count = [1] * Number_of_Bandits   # Array for Number of bandits try times, e.g. [1, 1, 1, 1]
Thompson_win_count = [0] * Number_of_Bandits  # Array for Number of bandits win times, e.g. [0, 0, 0, 0]
Thompson_loss_count = [0] * Number_of_Bandits  # Array for Number of bandits loss times, e.g. [0, 0, 0, 0]
Thompson_average_reward = []

#   Initialize for UCB
ucb_runing_count = [1] * Number_of_Bandits
ucb_rewards = [0] * Number_of_Bandits
ucb_average_reward = []

#   Initialize for e-greedy
e_greedy_runing_count = [1] * Number_of_Bandits
e_greedy_rewards = [0] * Number_of_Bandits
e_greedy_average_reward = [] 

# Thompson sampling for the the multi-armed bandits
for step in range(1, N):
    # Beta distribution and alfa beta calculation
    bandit_distribution = []

    # We calculate the main equation (beta distribution) using statistics library (note +1 for avoiding zero and undefined value)
    for i in range(len(Thompson_runing_count)):
        bandit_distribution.append(stats.beta(a = Thompson_win_count[i] + 1, b = Thompson_loss_count[i] + 1)) 
        
    '''
    Or we can write in following form using zip more compactly
    for run_count, win_count in zip(bandit_runing_count, bandit_win_count): 
        # create a tuple() of count and win, uses zip() to iterate over bandit_runing_count and bandit_win_count simultaneously.
        bandit_distribution.append (stats.beta(a = win_count + 1, b = run_count - win_count + 1)) 
    '''

    prob_theta_samples = []
    # Theta probability sampeling for each bandit
    for p in bandit_distribution:
        prob_theta_samples.append(p.rvs(1)) #rvs(random variates sampling) method provides random samples of distibution

    # Select best bandit based on theta sample a bandit
    select_bandit = np.argmax(prob_theta_samples)

    # Run bandit and update win count, loss count, and run count
    if(bandit_run(select_bandit) == 1):
        Thompson_win_count[select_bandit] += 1 
    else:
        Thompson_loss_count[select_bandit] += 1 

    Thompson_runing_count[select_bandit] += 1

    if step == 3 or step == 11 or (step % 100 == 1 and step <= 1000) :
        Thompson_plot(bandit_distribution, step - 1, next(ax))
    
    Thompson_reward_list = []
    for i in range(len(Thompson_runing_count)): 
        Thompson_reward_list.append(Thompson_win_count[i] / Thompson_runing_count[i])    # We calculte bandit average (Probability of Winning)

    # Or we can write in following form using zip more compactly
    # It does elemtwise division of lists using zip() and create new list [AvgRewardARM1, AvgRewardARM2, AvgRewardARM3, ...]
    # average_reward_list = ([i / j for i, j in zip(bandit_win_count, bandit_runing_count)])

    # Get average of all bandits into only one reward value
    #averaged_total_reward = 0
    '''
    for avged_arm_reward in (average_reward_list):
        averaged_total_reward += avged_arm_reward       #adds the avged_arm_reward to averaged_total_reward in each iteration
    average_reward.append(averaged_total_reward)        #append average_reward into averged_total_reward
    '''
    
    # Or we can write in following form using zip more compactly
    Thompson_average_reward.append(sum(Thompson_reward_list))

    # averaged_total_reward = average_reward_list[0] # to show wining chance of one machine


    '''----------------------------------------------------------UCB algorithm---------------------------------------------------------------'''
    if step <= Number_of_Bandits:
        ucb_bandit = step - 1
    else:
        for i in range(Number_of_Bandits):
            ucb_values = [ucb_rewards[i] / ucb_runing_count[i] + np.sqrt(2*np.log(step) / ucb_runing_count[i])
                          if ucb_runing_count[i] > 0 else float('inf')]
        ucb_bandit = np.argmax(ucb_values)

    ucb_reward = bandit_run(ucb_bandit)
    ucb_runing_count[ucb_bandit] += 1
    ucb_rewards[ucb_bandit] += ucb_reward

    # Calculate average reward for UCB
    ucb_avg_rewards = [ucb_rewards[i] / ucb_runing_count[i] if ucb_runing_count[i] > 0 else 0 for i in range(Number_of_Bandits)]
    ucb_average_reward.append(sum(ucb_avg_rewards) / Number_of_Bandits)

    '''-------------------------------------------------------e-greedy Algorithm-------------------------------------------------------------'''
    if np.random.rand() < epsilon:
        e_greedy_bandit = np.random.randint(0, Number_of_Bandits)
    else:
        for i in range(Number_of_Bandits):
            estimated_values = [e_greedy_rewards[i] / e_greedy_runing_count[i] 
                                if e_greedy_runing_count[i] > 0 else 0]
        e_greedy_bandit = np.argmax(estimated_values)

    e_greedy_reward = bandit_run(e_greedy_bandit)
    e_greedy_runing_count[e_greedy_bandit] += 1
    e_greedy_rewards[e_greedy_bandit] += e_greedy_reward

    #Calculate average reward for e-greedy
    for i in range(Number_of_Bandits):
        e_greedy_avg_rewards = [e_greedy_rewards[i] / e_greedy_runing_count[i] 
                                if e_greedy_runing_count[i] > 0 else 0]
    e_greedy_average_reward.append(sum(e_greedy_avg_rewards) / Number_of_Bandits)





plt.tight_layout()
plt.show()

plot_rewards(Thompson_average_reward, ucb_average_reward, e_greedy_average_reward)
