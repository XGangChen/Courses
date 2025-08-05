from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

# Random Tumor size, Tumor location, and Patient age with some noise on lables
np.random.seed(0)
X = np.random.rand(150, 3) * [100, 10, 80]  # Scale data to appropriate ranges
Y = np.where(X[:, 2] > 35, 1, -1) # if the age is greater than 35 it is Tumor
mean = 0 # The mean for the Gaussian distribution
std_dev = 0.5 # THe standard deviation for the Gaussian distribution (increased from 0.3 to 0.5)
noise_Y = np.random.normal(mean, std_dev, Y.shape) # We generate a gaussian noise with the same shape as Y
Y_nois = np.round(Y + noise_Y) # we van add the noise to the original Y and round the values
Y_nois = np.clip(Y_nois, -1, 1) # To ensure after adding noise  the values are still in the range of -1 and 1

# Fit SVM model
classifier = svm.SVC(kernel='linear', C=1.0) # poly  rbf   sigmoid  precomputed # note to plot and visualize need seperate impementations
classifier.fit(X, Y_nois)  # Use Y_noisy instead of Y

# Get intercept and bias of the hyperplane
b = classifier.intercept_[0]
w = classifier.coef_[0]

# 5 folds Ccross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X, Y_nois, cv=5, scoring='accuracy') # scoring= accuracy precision  recall f1_macro f1 roc_auc average_precision

print(scores)

# Plot the hyperplane
plot_x, plot_y = np.meshgrid(np.linspace(0, 100, 10), np.linspace(0, 10, 10)) # evenly spaced values between 0 and 100 for x and between 0 and 10 for y
plot_z = (-w[0] * plot_x - w[1] * plot_y - b) / w[2]

colors = np.where(Y_nois > 0, 'red', 'green')  # Use Y_noisy instead of Y

# Plot data and hyperplane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors)
ax.plot_surface(plot_x, plot_y, plot_z, alpha=0.5)
ax.set_xlabel('Tumor size')
ax.set_ylabel('Tumor location')
ax.set_zlabel('Patient age')
plt.show()
