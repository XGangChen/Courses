from scipy.stats import t, linregress
import numpy as np
import matplotlib.pyplot as plt

# Sample data
sample = [96.2, 97.5, 98.1, 98.7, 99.3, 100.1, 100.9, 101.2, 102.5, 103.8,
          104.3, 105.1, 105.8, 106.4, 107.2, 107.8, 108.4, 108.9, 109.2, 109.7]

# We Calculate the simple linear regression model
# **Use slope, and intercept and calculate  T-statistic, and p_value yourself with course slides**
slope, intercept, r_square, p_value, std_err = linregress(np.arange(len(sample)), sample)

# Print the slope and intercept
print("Slope:", slope)
print("Intercept:", intercept)

# Determine the Degrees of freedom
df = len(sample) - 2

# T-statistic
t_statistic = 1.5 #### something random (you calculate)
print("t-statistic: ", t_statistic)

# One-tailed p-value
if t_statistic < 0:
    p_val = t.cdf(t_statistic, df=df) # t.cdf: cumulative distribution function (CDF) of the Student's t-distribution
else:
    p_val = 1 - t.cdf(t_statistic, df=df)
print("one-tailed p-value: ", p_val)

if(p_val < 0.05):
    print('Reject the null hypothesis') 
else:
    print('Accept the null hypothesis') 

# Two-tailed p-value
if t_statistic < 0:
    p_val = t.cdf(t_statistic, df=df) * 2
else:
    p_val = (1 - t.cdf(t_statistic, df=df)) * 2
print("two-tailed p-value: ", p_val)

if(p_val < 0.05):
    print('Reject the null hypothesis') 
else:
    print('Accept the null hypothesis') 

# Plot the data and regression line
plt.scatter(np.arange(len(sample)), sample, color='black')
plt.plot(np.arange(len(sample)), intercept + slope*np.arange(len(sample)), color='green')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
