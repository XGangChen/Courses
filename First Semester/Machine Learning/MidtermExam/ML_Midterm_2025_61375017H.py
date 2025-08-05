import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold
import statsmodels.api as sm
from scipy.stats import f

data = pd.DataFrame({
    'SocialMedia': [
        340.1, 154.5, 127.2, 261.5, 290.8, 115.7, 167.5, 230.2, 115.6, 309.8,
        176.1, 324.7, 130.8, 207.5, 314.1, 305.4, 177.8, 391.4, 179.2, 257.3,
        374.11, 169.95, 139.92, 287.65, 319.88, 127.27, 184.25, 253.22, 127.16, 340.78,
        193.71, 357.17, 143.88, 228.25, 345.51, 335.94, 195.58, 430.54, 197.12, 283.03,
        323.095, 146.775, 120.84, 248.425, 276.26, 109.915, 159.125, 218.69, 109.82, 294.31,
        167.295, 308.465, 124.26, 197.125, 298.395, 290.13, 168.91, 372.83, 170.24, 244.435
    ],
    'Billboard': [
        169.2, 145.1, 169.3, 157.5, 157.4, 182.0, 130.5, 118.6, 108.0, 128.2,
        123.2, 103.0, 164.9, 106.2, 145.0, 151.9, 213.0, 154.8, 117.3, 126.1,
        177.66, 152.355, 177.765, 165.375, 165.27, 191.1, 137.025, 124.53, 113.4, 134.61,
        129.36, 108.15, 173.145, 111.51, 152.25, 159.495, 223.65, 162.54, 123.165, 132.405,
        152.28, 130.59, 152.37, 141.75, 141.66, 163.8, 117.45, 106.74, 97.2, 115.38,
        110.88, 92.7, 148.41, 95.58, 130.5, 137.79, 193.05, 140.82, 106.57, 114.645
    ],
    'Sales': [
        29.1, 17.4, 16.3, 25.5, 19.9, 14.2, 18.8, 20.2, 11.8, 18.6,
        16.6, 23.4, 15.2, 15.7, 26.0, 29.4, 19.5, 31.4, 18.3, 21.6,
        30.371015, 17.714363, 17.538445, 27.397403, 20.80007, 14.751211, 20.826538, 21.55986, 13.139398, 19.204558,
        16.979135, 23.812623, 16.345842, 17.244362, 27.167014, 30.092623, 21.099325, 33.133781, 18.751607, 22.305827,
        27.667176, 15.608117, 15.723191, 24.124267, 19.01121, 13.662289, 18.241651, 19.730123, 10.968067, 17.680196,
        15.778138, 21.662496, 14.495963, 14.460843, 24.996702, 27.738864, 18.649246, 29.420281, 17.243353, 20.276602
    ]
})

# Features and target
X = data[['SocialMedia', 'Billboard']].values
y = data['Sales'].values

# Hyperparameters
learning_rate = 1e-5
iterations = 200
batch_size = 4


kf = KFold(n_splits=2)

def minibatch_SGD(X_train, y_train, learning_rate, iterations, batch_size):
    n_samples, n_features = X_train.shape
    theta = np.zeros(n_features + 1)
    losses = []

    for it in range(iterations):
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            xb = X_shuffled[start_idx:end_idx]
            yb = y_shuffled[start_idx:end_idx]

            xb_bias = np.c_[np.ones(xb.shape[0]), xb]
            preds = xb_bias @ theta
            errors = preds - yb

            grad = (1 / xb.shape[0]) * (xb_bias.T @ errors)
            theta -= learning_rate * grad

        all_preds = np.c_[np.ones(n_samples), X_train] @ theta
        loss = np.mean((all_preds - y_train) ** 2)
        losses.append(loss)

    return theta, losses


mse_list = []
first_loss_history = None

for i, (train_index, val_index) in enumerate(kf.split(X)):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    theta, loss_history = minibatch_SGD(X_train, y_train, learning_rate, iterations, batch_size)

    X_val_bias = np.c_[np.ones(X_val.shape[0]), X_val]
    y_pred = X_val_bias @ theta
    mse = np.mean((y_val - y_pred) ** 2)
    mse_list.append(mse)

    # Save the loss history from the first fold only for plotting
    if i == 0:
        first_loss_history = loss_history

# Plot MSE per iteration
plt.figure(figsize=(8, 5))
plt.plot(range(1, iterations + 1), first_loss_history, marker='o', linestyle='-', color='orange')
plt.title("SGD Training MSE Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Final MSE
avg_mse = np.mean(mse_list)
print(f"Average Validation MSE: {avg_mse:.4f}")


# -------------------- F-statistic --------------------------
n = 60  # Sample size
k = 6   # Number of independent variables

# Fit OLS regression model
model = sm.OLS(y, X).fit()

# Get F-statistic from model summary
f_stat = model.fvalue
print("Calculated F-statistic:", round(f_stat, 4))

# Degrees of freedom
df1 = k                 # Numerator df (number of predictors)
df2 = n - k - 1         # Denominator df (residual degrees of freedom)

# Set significance level (alpha)
alpha = 0.05

# Calculate critical value from F-distribution
f_critical = f.ppf(1 - alpha, dfn=df1, dfd=df2)
print(f"Critical F-value(k=6) at alpha={alpha}: {round(f_critical, 4)}")

# ---------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data['SocialMedia'], data['Billboard'], data['Sales'], s=60)

ax.set_xlabel('Social Media Spend')
ax.set_ylabel('Billboard Spend')
ax.set_zlabel('Sales')
ax.set_title('3D Scatter Plot of Advertising Spend vs Sales')

plt.show()

# --------------------- R-squared ------------------------------
X = X.reshape(120, 1)

# calculate the slope and intercept of the regression line
numerator = np.sum((X - np.mean(X)) * (y - np.mean(y)))
denominator = np.sum((X - np.mean(X)) ** 2)
slope = numerator / denominator
intercept = np.mean(y) - slope * np.mean(X)

# predict sales based on advertising spend
y_pred = intercept + slope * X

# calculate the RSS
RSS = np.sum((y - y_pred) ** 2)
TSS = np.sum((y - np.mean(y)) ** 2)
print('RSS:', RSS)
print('TSS:', TSS)

R_sq = 1 - RSS/TSS

# R squared value ranges between 0 to 1 with higher values indicating a better fit, but my R_sq doens't fit cause the RSS is much bigger than TSS.
# I think there is something wrong in the data structure and calculate process.
print('R squard:', R_sq)

