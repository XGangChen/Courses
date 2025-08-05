import pandas as pd
import numpy as np

# a sample dataset with  advertising and sales
data = pd.DataFrame({
    'SocialMedia': [340.1, 154.5, 127.2], 
    'Sales': [29.1, 17.4, 16.3]
})

# Reshape the Numpy array into  a two_dimentional array with a single column
x = data['SocialMedia'].values.reshape(-1,1)
y = data['Sales'].values

# Slope and intercept of the regression line
numerator = np.sum((x - np.mean(x)) * (y - np.mean(y)))
denominator = np.sum((x - np.mean(x))** 2)
slope = numerator / denominator
intercept = np.mean(y) - slope * np.mean(x)

# Predict the sales based on the advertising spend
y_pred = intercept + slope * x

# Calculate the RSS
RSS = np.sum((y - y_pred) ** 2)
print(RSS)