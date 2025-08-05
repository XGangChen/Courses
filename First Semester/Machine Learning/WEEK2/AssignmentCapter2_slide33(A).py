import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed()

# Generate house sizes (sqft)
sizes = np.random.normal(loc=1500, scale=300, size=100)

# Generate house prices with a linear relationship + noise
price_per_sqft = 200
noise = np.random.normal(0, 20000, 100)
prices = sizes * price_per_sqft + noise

# Create DataFrame
df = pd.DataFrame({'Size (sqft)': sizes, 'Price ($)': prices})

# Linear regression using numpy
coefficients = np.polyfit(sizes, prices, deg=1)
regression_fn = np.poly1d(coefficients)

# Create predicted prices using the regression function
predicted_prices = regression_fn(sizes)

# Plot the scatter plot and regression line
plt.figure(figsize=(10, 6))
plt.scatter(sizes, prices, color='blue', alpha=0.6, label='Data Points')
plt.plot(sizes, predicted_prices, color='red', linewidth=2, label='Regression Line')
plt.title("House Price vs Size with Regression Line")
plt.xlabel("Size (sqft)")
plt.ylabel("Price ($)")
plt.grid(True)
plt.legend()
plt.show()
