import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# create some example data
x = np.array([1, 4, 6, 4, 5])
y = np.array([3, 5, 7, 9, 11])

# Linear regression objects
reg1 = LinearRegression()
reg2 = LinearRegression()

# Fit data to the regression models
reg1.fit(x.reshape(-1, 1), y)
reg2.fit(x[:3].reshape(-1, 1), y[:3]) # fit only to 3 points (incompleate)

# train predictions on the data for both models
y_pred1 = reg1.predict(x.reshape(-1, 1))
y_pred2 = reg2.predict(x.reshape(-1, 1))

#----------------Implementation similar to class slides math-------------
# MSE calculation
def calculate_mse(y, y_pred):
    n = len(y)
    return np.sum(pow((y - y_pred),2)) / n

# RSE calculation
def calculate_rse(y, y_pred):
    mse = calculate_mse(y, y_pred)
    return np.sqrt(mse)

# R-squared calculation
def calculate_r_squared(y, y_pred):
    tss = np.sum(pow((y - np.mean(y)),2))
    rss = np.sum(pow((y - y_pred),2))
    return 1 - (rss / tss)

# TSS calculation
def calculate_tss(y):
    return np.sum(pow((y - np.mean(y)),2))

# F-statistic calculation
def calculate_f_statistic(y, y_pred):
    mse = calculate_mse(y, y_pred)
    tss = calculate_tss(y)
    df_regression = 1
    df_residuals = len(y) - 2
    msr = (tss - np.sum(pow((y - y_pred),2))) / df_regression
    f_stat = (msr / mse) / df_residuals
    return f_stat

# calculate the metrics for both models
mse1 = calculate_mse(y, y_pred1)
rse1 = calculate_rse(y, y_pred1)
r_squared1 = calculate_r_squared(y, y_pred1)
tss1 = calculate_tss(y)
f_statistic1 = calculate_f_statistic(y, y_pred1)

mse2 = calculate_mse(y, y_pred2)
rse2 = calculate_rse(y, y_pred2)
r_squared2 = calculate_r_squared(y, y_pred2)
tss2 = calculate_tss(y)
f_statistic2 = calculate_f_statistic(y, y_pred2)

# print the results
print("Regression 1 (true): ")
print("MSE:", mse1)
print("RSE:", rse1)
print("TSS:", tss1)
print("R-squared:", r_squared1)
print("F-statistic:", f_statistic1)

print('***************')
print("Regression 2 (Incomplete): ")
print("MSE:", mse2)
print("RSE:", rse2)
print("TSS:", tss2)
print("R-squared:", r_squared2)
print("F-statistic:", f_statistic2)

# plot the data and the regression lines
plt.scatter(x, y)
plt.plot(x, y_pred1, label="Regression 1 (true)")
plt.plot(x, y_pred2, label="Regression 2 (Incomplete)")
plt.legend()
plt.show()
