import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import tree

# Prepare the data
data = pd.DataFrame({
    'SocialMedia': [340.1, 154.5, 127.2, 261.5, 290.8, 115.7, 167.5, 230.2, 115.6, 309.8, 176.1, 324.7, 130.8, 207.5, 314.1, 305.4, 177.8, 391.4, 179.2, 257.3],
    'TV': [147.8, 150.3, 157.9, 151.3, 133.8, 156.9, 140.8, 127.6, 109.1, 109.6, 112.8, 131.0, 142.1, 130.6, 140.9, 170.7, 149.6, 153.6, 130.5, 130.9],
    'Billboard': [169.2, 145.1, 169.3, 157.5, 157.4, 182.0, 130.5, 118.6, 108.0, 128.2, 123.2, 103.0, 164.9, 106.2, 145.0, 151.9, 213.0, 154.8, 117.3, 126.1],
    'Sales': [29.1, 17.4, 16.3, 25.5, 19.9, 14.2, 18.8, 20.2, 11.8, 18.6, 16.6, 23.4, 15.2, 15.7, 26.0, 29.4, 19.5, 31.4, 18.3, 21.6],
    'Class': ['Medium', 'Low', 'Low', 'Medium', 'Medium', 'Low', 'Medium', 'Low', 'Low', 'Medium', 'Low', 'Medium', 'Low', 'Low', 'Medium', 'Medium', 'Low', 'Medium', 'Medium', 'Low']
})

X = data[['SocialMedia', 'TV', 'Billboard']]
y_regression = data['Sales']
y_classification = data['Class']

regressor = RandomForestRegressor(n_estimators=5, random_state=10) # n_estimators is number of trees also. random_state=42 is random seed for the random number generator
classifier = RandomForestClassifier(n_estimators=120, random_state=10)

regressor.fit(X, y_regression)
classifier.fit(X, y_classification)

# Get the unique class labels
my_class_labels = classifier.classes_

# Plot the Random Forest for regression
fig1, axes1 = plt.subplots(nrows=1, ncols=len(regressor.estimators_), figsize=(20, 10))
for i, estimator in enumerate(regressor.estimators_):
    tree.plot_tree(estimator, filled=True, ax=axes1[i], fontsize=4)
    axes1[i].set_title('Random forest for Regression {}'.format(i+1))

# Plot the Random Forest for classification
# fig2, axes2 = plt.subplots(nrows=1, ncols=len(classifier.estimators_), figsize=(20, 10))
# for i, estimator in enumerate(classifier.estimators_):
#     tree.plot_tree(estimator, filled=True, ax=axes2[i], fontsize=7, class_names=my_class_labels)
#     axes2[i].set_title('Decision Tree - Classification {}'.format(i+1))

plt.show()
