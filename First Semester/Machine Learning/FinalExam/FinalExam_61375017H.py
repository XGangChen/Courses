import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
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
    ], 
    'Class': ['Medium', 'Low', 'Low', 'Medium', 'Medium', 'Low', 'Medium', 'Low', 'Low', 'Medium', 'Low', 'Medium', 'Low', 'Low', 'Medium', 'Medium', 'Low', 'Medium', 'Medium', 'Low', 
              'Medium', 'Low', 'Low', 'Medium', 'Medium', 'Low', 'Medium', 'Low', 'Low', 'Medium', 'Low', 'Medium', 'Low', 'Low', 'Medium', 'Medium', 'Low', 'Medium', 'Medium', 'Low', 
              'Medium', 'Low', 'Low', 'Medium', 'Medium', 'Low', 'Medium', 'Low', 'Low', 'Medium', 'Low', 'Medium', 'Low', 'Low', 'Medium', 'Medium', 'Low', 'Medium', 'Medium', 'Low']
})

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data['SocialMedia'], data['Billboard'], data['Sales'], s=60)

ax.set_xlabel('Social Media Spend')
ax.set_ylabel('Billboard Spend')
ax.set_zlabel('Sales')
ax.set_title('3D Scatter Plot of Advertising Spend vs Sales')
plt.show()

X = data[['SocialMedia', 'Billboard', 'Sales']]
y_regression = data['SocialMedia']
y_classification = data['Class']

regressor = DecisionTreeRegressor(max_depth=10, random_state=42, min_samples_leaf=2, min_samples_split=8, max_features='log2')
classifier = DecisionTreeClassifier(criterion='gini')

regressor.fit(X, y_regression)
classifier.fit(X, y_classification)

my_class_labels = classifier.classes_ # class labels

# We Plot the decision tree for regression
fig1, ax1 = plt.subplots(figsize=(12, 8))
tree.plot_tree(regressor, filled=True, ax=ax1, fontsize=7)
ax1.set_title('Decision Tree for Regression Test')

# We Plot the decision tree for classification
fig2, ax2 = plt.subplots(figsize=(12, 8))
tree.plot_tree(classifier, filled=True, ax=ax2, fontsize=7, class_names=my_class_labels)
ax2.set_title('Decision Tree for Classification Task')

plt.show()

avg_mse = k_fold_cv(X, y_regression, k=5)
print("Average MSE:", avg_mse)


