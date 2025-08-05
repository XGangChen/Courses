import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
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

regressor = DecisionTreeRegressor()
classifier = DecisionTreeClassifier(criterion='gini')

regressor.fit(X, y_regression)
classifier.fit(X, y_classification)

my_class_labels = classifier.classes_ # class labels

# We Plot the decision tree for regression
fig1, ax1 = plt.subplots(figsize=(12, 8))
tree.plot_tree(regressor, filled=True, ax=ax1, fontsize=7)
ax1.set_title('Decision Tree for Regression Tast')

# We Plot the decision tree for classification
fig2, ax2 = plt.subplots(figsize=(12, 8))
tree.plot_tree(classifier, filled=True, ax=ax2, fontsize=7, class_names=my_class_labels)
ax2.set_title('Decision Tree for Classification Task')

plt.show()
