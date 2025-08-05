import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# a sample dataset with advertising and sales
data = pd.DataFrame({
    'SocialMedia': [340.1, 154.5, 127.2, 261.5, 290.8, 115.7, 167.5, 230.2, 115.6, 309.8, 176.1, 324.7, 130.8, 207.5, 314.1, 305.4, 177.8, 391.4, 179.2, 257.3],
    'TV': [147.8, 150.3, 157.9, 151.3, 133.8, 156.9, 140.8, 127.6, 109.1, 109.6, 112.8, 131.0, 142.1, 130.6, 140.9, 170.7, 149.6, 153.6, 130.5, 130.9],
    'Billboard': [169.2, 145.1, 169.3, 157.5, 157.4, 182.0, 130.5, 118.6, 108.0, 128.2, 123.2, 103.0, 164.9, 106.2, 145.0, 151.9, 213.0, 154.8, 117.3, 126.1],
    'Sales': [29.1, 17.4, 16.3, 25.5, 19.9, 14.2, 18.8, 20.2, 11.8, 18.6, 16.6, 23.4, 15.2, 15.7, 26.0, 29.4, 19.5, 31.4, 18.3, 21.6]
})
# reshape the Numpy array into a two-dimensional array with a single column
x = data['SocialMedia'].values.reshape(-1, 1)
y = data['Sales'].values

# calculate the slope and intercept of the regression line
numerator = np.sum((x - np.mean(x)) * (y - np.mean(y)))
denominator = np.sum((x - np.mean(x)) ** 2)
slope = numerator / denominator
intercept = np.mean(y) - slope * np.mean(x)

# predict sales based on advertising spend
y_pred = intercept + slope * x
# calculate the RSS
RSS = np.sum((y - y_pred) ** 2)

# plot the data and regression line
plt.scatter(x, y, color='darkblue')
plt.plot(x, y_pred, color='red', alpha=0.5)

# print the slope, intercept, and RSS
plt.text(x.min()+0.25*(x.max()-x.min()), y.max()-0.25*(y.max()-y.min()), f"Slope: {slope:.2f}\nIntercept: {intercept:.2f}\nRSS: {RSS:.2f}", fontsize=8)

# set the title and axis labels
plt.title('Advertising (SocialMedia)')
plt.xlabel('Advertising Spend (1000$)')
plt.ylabel('Sales')

# show the plot
plt.show()