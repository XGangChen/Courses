import numpy as np
import matplotlib.pyplot as plt

# Define the grid
x1 = np.linspace(-1.5, 1.5, 400)
x2 = np.linspace(-1.5, 1.5, 400)
X1, X2 = np.meshgrid(x1, x2)

# Define V and V_dot
V = 0.5 * X1**2 + 0.5 * X2**2 + 0.25 * X1**4

# Define dot{x1}
dot_x1 = (X1 * X2 - 1) * X1**3 + (X1 * X2 - 1 + X2**2) * X1
# Compute dot{V}
V_dot = (X1 + X1**3) * dot_x1 - X2**2

# Plot
plt.figure(figsize=(7,6))
# Lyapunov level sets
plt.contour(X1, X2, V, levels=[0.05, 0.1, 0.2, 0.4], colors='blue')
# Where V_dot = 0
plt.contour(X1, X2, V_dot, levels=[0], colors='red')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('New ROA Estimate (Red: $\dot{V}=0$, Blue: $V(x)=c$)')
plt.grid(True)
plt.axis('equal')
plt.show()
