# 題目 Q1: Logistic Regression using Gradient Descent
# (a) 從零開始實作 logistic regression（不能使用 sklearn）
# (b) 繪製 cost function (log-loss) 隨迭代的變化
# (c) 使用模型預測五筆測試資料，並說明預測結果

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    return -(1/m) * np.sum(y * np.log(h + 1e-5) + (1 - y) * np.log(1 - h + 1e-5))

def logistic_regression(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    theta = np.zeros((n, 1))
    costs = []

    for _ in range(epochs):
        h = sigmoid(X @ theta)
        gradient = (X.T @ (h - y)) / m
        theta -= lr * gradient
        costs.append(compute_cost(X, y, theta))
    return theta, costs

# 產生簡單資料
X = np.array([[1, 2], [1, 3], [1, 5], [1, 6], [1, 8]])
y = np.array([[0], [0], [1], [1], [1]])

# 訓練模型
theta, costs = logistic_regression(X, y)

# 繪製 loss 曲線
plt.plot(costs)
plt.title("Log-Loss over Iterations")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# 測試預測
X_test = np.array([[1, 2], [1, 4], [1, 5.5], [1, 7], [1, 9]])
preds = sigmoid(X_test @ theta)
print("Predicted Probabilities:", preds.ravel())
print("Predicted Classes:", (preds > 0.5).astype(int).ravel())