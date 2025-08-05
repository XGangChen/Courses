from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
import matplotlib.pyplot as plt

# Function to solve a problem
def solve_problem(problem_name, n_var):
    # Get the problem
    problem = get_problem(problem_name, n_var=n_var)
    
    # Set up the NSGA-II algorithm
    algorithm = NSGA2(pop_size=100)
    
    # Perform optimization (limiting to 25000 evaluations)
    res = minimize(
        problem,
        algorithm,
        ('n_eval', 25000),  # Stopping criteria based on evaluations
        seed=1,
        verbose=True
    )
    
    return res.F

# Solve all supported ZDT problems
zdt_results = {
    "ZDT1": solve_problem("zdt1", n_var=30),  # ZDT1 with 30 variables
    "ZDT2": solve_problem("zdt2", n_var=30),  # ZDT2 with 30 variables
    "ZDT3": solve_problem("zdt3", n_var=30),  # ZDT3 with 30 variables
    "ZDT4": solve_problem("zdt4", n_var=10),  # ZDT4 with 10 variables
    "ZDT6": solve_problem("zdt6", n_var=10),  # ZDT6 with 10 variables
}

# Plot results in subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Iterate through the ZDT results and plot
for i, (zdt_name, results) in enumerate(zdt_results.items()):
    ax = axes[i // 3, i % 3]
    ax.scatter(results[:, 0], results[:, 1], label=zdt_name)
    ax.set_title(zdt_name)
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.legend()
    ax.grid()

# Remove empty subplot if present
axes[1, 2].axis("off")

# Adjust layout
plt.tight_layout()
plt.show()
