import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create a dataset with 100 samples
np.random.seed()
num_samples = 100

# Generate semi-random house size (in squarte feet)
sizes = np.random.normal(loc=1500, scale=300, size=num_samples)

# Generate location coordinates in a circular area
x_coords = np.random.uniform(low=-10, high=10, size=num_samples)
y_coords = np.random.uniform(low=-10, high=10, size=num_samples)

# Compute the distance from the center of the circle
distances = np.sqrt(x_coords**2 + y_coords**2)

# Create a meaningful  relationship between size and price (price increase with size)
# Add some randonm noise to the price to simulate real-world variablity
price_per_sqft = 200    # base price per square foot
location_influence = distances * 8000
noise = np.random.normal(loc=0, scale=20000, size=num_samples)
prices = sizes * price_per_sqft - 2 * location_influence + noise

#Create DataFrame
df = pd.DataFrame({
    'Size (sqft)': sizes, 
    'X': x_coords,
    'Y': y_coords,
    'Distance from Downtown': distances,
    'Price ($)': prices
    })

# plot 1: Size vs Price
plt.figure(figsize=(10, 6))
plt.scatter(df['Size (sqft)'], df['Price ($)'], alpha=0.6, label='Data Points')
coeff_size = np.polyfit(df['Size (sqft)'], df['Price ($)'], 1)
reg_size = np.poly1d(coeff_size)
plt.plot(df['Size (sqft)'], reg_size(df['Size (sqft)']), color='red', label='Regression Line')
plt.title('Size vs Price')
plt.xlabel('Size (sqft)')
plt.ylabel('Price ($)')
plt.grid(True)
plt.legend()
plt.show()

# plot 2: Distance from Downtown vs Price
plt.figure(figsize=(10, 6))
plt.scatter(df['Distance from Downtown'], df['Price ($)'], alpha=0.6, label='Data Points')
coeff_dist = np.polyfit(df['Distance from Downtown'], df['Price ($)'], 1)
reg_dist = np.poly1d(coeff_dist)
plt.plot(df['Distance from Downtown'], reg_dist(df['Distance from Downtown']), color='green', label='Regression Line')
plt.title('Distance from Downtown vs Price')
plt.xlabel('Distance from Downtown')
plt.ylabel('Price ($)')
plt.grid(True)
plt.legend()
plt.show()


