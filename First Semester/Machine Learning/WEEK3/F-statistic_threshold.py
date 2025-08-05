import numpy as np
import statsmodels.api as sm
from scipy.stats import f

# ----------------------------
# Simulated data
np.random.seed(42)
n = 30  # Sample size
k = 3   # Number of independent variables

# Generate random independent variables
X = np.random.randn(n, k)
X = sm.add_constant(X)  # Add intercept

# Generate coefficients and dependent variable y
beta = np.array([2, 0.5, -1, 0.7])  # 3 predictors + intercept
epsilon = np.random.randn(n)
y = X @ beta + epsilon

# Fit OLS regression model
model = sm.OLS(y, X).fit()

# Get F-statistic from model summary
f_stat = model.fvalue
print("Calculated F-statistic:", round(f_stat, 4))

# Degrees of freedom
df1 = k                 # Numerator df (number of predictors)
df2 = n - k - 1         # Denominator df (residual degrees of freedom)

# Set significance level (alpha)
alpha = 0.05

# Calculate critical value from F-distribution
f_critical = f.ppf(1 - alpha, dfn=df1, dfd=df2)
print(f"Critical F-value at alpha={alpha}: {round(f_critical, 4)}")

# ----------------------------
# Conclusion
if f_stat > f_critical:
    print("Reject the null hypothesis: At least one independent variable is significant.")
else:
    print("Fail to reject the null hypothesis: No evidence that predictors are significant.")
