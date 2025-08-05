from scipy.stats import t

# Example input values (you can replace these with your own values)
t_statistic = -2.1  # Example t-statistic
df = 20             # Degrees of freedom

# One-tailed p-value
if t_statistic < 0:
    p_val_one_tailed = t.cdf(t_statistic, df=df)
else:
    p_val_one_tailed = 1 - t.cdf(t_statistic, df=df)

print("One-tailed p-value:", p_val_one_tailed)

# Two-tailed p-value
if t_statistic < 0:
    p_val_two_tailed = t.cdf(t_statistic, df=df) * 2
else:
    p_val_two_tailed = (1 - t.cdf(t_statistic, df=df)) * 2

print("Two-tailed p-value:", p_val_two_tailed)
