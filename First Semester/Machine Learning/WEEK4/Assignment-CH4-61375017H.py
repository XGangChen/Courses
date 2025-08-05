import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Generate synthetic data
np.random.seed(42)
n = 100
# 'social_media' & 'tv' are 1-dimensional NumPy arrays with 100 elements.
# Each element is a floating-point number uniformly distributed between 0 and 100.
social_media = np.random.uniform(0, 100, size=n)
tv = np.random.uniform(0, 100, size=n)

# Synergy effect (interaction term)
interaction = social_media * tv

# Sales data for Product A and Product B
# Assume socil media has more weight, but interaction boots both
sales_A = 0.5 * social_media + 0.3 * tv + 0.02 * interaction + np.random.normal(0, 10, size=n)
sales_B = 0.4 * social_media + 0.4 * tv + 0.015 * interaction + np.random.normal(0, 10, size=n)

# Create a DataFrame
df = pd.DataFrame({
    'Social_media': social_media,
    'Tv': tv,
    'Interaction': interaction,
    'Sales_A': sales_A,
    'Sales_B': sales_B
})

# Prepare input/output for multivariate regression
X = df[['Social_media', 'Tv', 'Interaction']]
Y = df[['Sales_A', 'Sales_B']]

# Fit multivariate multiple regression model
model = LinearRegression()
model.fit(X, Y)

# Show coefficients (interpret synergy via interaction term)
coef_df = pd.DataFrame(model.coef_, columns=['Social_media', 'Tv', 'Interaction'], index=['Sales_A', 'Sales_B'])
print("Regression coefficients:\n", coef_df)

# Predict and plot actual vs. predicted
perdicted = model.predict(X)
df['Predicted_A'] = perdicted[:, 0]
df['Predicted_B'] = perdicted[:, 1]

# Visualize Actual vs. Predicted Sales
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(df['Sales_A'], df['Predicted_A'], alpha=0.7)
plt.plot([df['Sales_A'].min(), df['Sales_A'].max()], 
         [df['Sales_A'].min(), df['Sales_A'].max()], 'r--')
plt.xlabel('Actual Sales A')
plt.ylabel('Predicted Sales A')
plt.title('Product A - Actual vs. Predicted')

plt.subplot(1, 2, 2)
plt.scatter(df['Sales_B'], df['Predicted_B'], alpha=0.7)
plt.plot([df['Sales_B'].min(), df['Sales_B'].max()], 
         [df['Sales_B'].min(), df['Sales_B'].max()], 'r--')
plt.xlabel('Actual Sales B')
plt.ylabel('Predicted Sales B')
plt.title('Product B - Actual vs. Predicted')

plt.tight_layout()
plt.show()

# # Combine dependent variables into one DataFrame using 'long' format
# df_long = pd.concat([
#     df[['Social_media', 'Tv', 'Interaction']].assign(Sales=df['Sales_A'], Product='A'), 
#     df[['Social_media', 'Tv', 'Interaction']].assign(Sales=df['Sales_B'], Product='B')
# ])

# # Fit a linear model with interaction between predictors and product type
# model = smf.ols('Sales ~ Social_media + Tv + Interaction + Product + Product:Social_media + Product:Tv + Product:Interaction', data=df_long).fit()

# #  Summary with t-stats and p-values
# print(model.summary())

# # simple correlation matrix
# print(df[['Sales_A', 'Sales_B']].corr())

# # From Residuals (after fitting)
# df_long['Predicted'] = model.predict(df_long)
# residuals = df_long['Sales'] - df_long['Predicted']
# # Create a column for residuals
# df_long['Residual'] = df_long['Sales'] - df_long['Predicted']

# # Pivot to get residuals of A and B side by side
# resid_wide = df_long.pivot(columns='Product', values='Residual')

# # Correlation between residuals of Sales_A and Sales_B
# print("Residual correlation between Product A and B:\n", resid_wide.corr())

# # Statistical Test
# from statsmodels.multivariate.manova import MANOVA

# # Combine into a single DataFrame for MANOVA
# manova_data = df[['Sales_A', 'Sales_B', 'Social_media', 'Tv', 'Interaction']]
# maov = MANOVA.from_formula('Sales_A + Sales_B ~ Social_media + Tv + Interaction', data=manova_data)
# print(maov.mv_test())