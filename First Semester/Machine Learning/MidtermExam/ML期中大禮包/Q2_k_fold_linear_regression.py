# 題目 Q2: K-Fold Cross Validation for Linear Regression
# (a) 實作不使用 sklearn 的線性回歸
# (b) 實作 K-Fold 交叉驗證（K=5），每折回傳 MSE
# (c) 回傳所有折的平均 MSE

import numpy as np
from sklearn.metrics import mean_squared_error

def linear_regression(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def k_fold_cv(X, y, k=5):
    fold_size = len(X) // k
    mse_list = []

    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size
        X_val, y_val = X[start:end], y[start:end]
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)

        theta = linear_regression(X_train, y_train)
        y_pred = X_val @ theta
        mse = mean_squared_error(y_val, y_pred)
        mse_list.append(mse)
    
    return np.mean(mse_list)

# 生成範例資料
X = np.hstack((np.ones((20, 1)), np.arange(20).reshape(-1, 1)))
y = 2 + 3 * X[:, 1:2] + np.random.randn(20, 1)

avg_mse = k_fold_cv(X, y, k=5)
print("Average MSE:", avg_mse)