import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt

# Extended dataset with 3 media types
np.random.seed(42)
data = pd.DataFrame({
    'SocialMedia': np.random.uniform(100, 350, 40),
    'TV': np.random.uniform(150, 400, 40),
    'Radio': np.random.uniform(30, 90, 40),
    'Sales': np.random.uniform(10, 30, 40)
})

# Features and target
X = data[['SocialMedia', 'TV', 'Radio']].values
y = data['Sales'].values

# Hyperparameters
learning_rate = 1e-5
iterations = 200
batch_size = 4
loo = LeaveOneOut()

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

# LOOCV loop
mse_list = []
first_loss_history = None

for i, (train_index, val_index) in enumerate(loo.split(X)):
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

# Plot MSE per iteration for the first LOOCV fold
plt.figure(figsize=(8, 5))
plt.plot(range(1, iterations + 1), first_loss_history, marker='o', linestyle='-', color='orange')
plt.title("SGD Training MSE Over Iterations (First LOOCV Fold)")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Final LOOCV MSE
avg_mse = np.mean(mse_list)
print(f"Average LOOCV Validation MSE: {avg_mse:.4f}")
