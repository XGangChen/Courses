import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Schwefel's P2.22 Function (Unimodal)
def schwefel_unimodal(x):
    schwefel_uni = np.sum(np.abs(x)) + np.prod(np.abs(x), axis=0)
    return schwefel_uni

# Schwefel's Function (Multimodal)
def schwefel_multimodal(x):
    d = len(x)
    schwefel_muti = 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=0)
    return schwefel_muti

# Nelder-Mead Optimization Function
def nelder_mead(fitness_function, dim, lower_bound, upper_bound, max_iterations, evaluation_limit=200000):
    # Initialize a random starting point within bounds
    x0 = np.random.uniform(lower_bound, upper_bound, dim)

    # History tracking
    fitness_history = []
    total_evaluations = [0]  # Use a mutable object to track evaluations inside the callback

    # Callback function to record the best fitness at each iteration
    def callback(xk):
        fitness_history.append(fitness_function(xk))

    # Wrapper to count evaluations
    def fitness_with_count(x):
        if total_evaluations[0] >= evaluation_limit:
            return float('inf')  # Stop the optimizer
        total_evaluations[0] += 1
        return fitness_function(x)

    # Perform optimization using Nelder-Mead
    result = minimize(
        fitness_with_count, 
        x0, 
        method='Nelder-Mead', 
        options={'maxiter': max_iterations, 'disp': True, 'adaptive': True},
        callback=callback
    )

    # Return results
    best_position = result.x
    best_fitness = result.fun
    return best_position, best_fitness, fitness_history, total_evaluations[0]

if __name__ == "__main__":
    num_dimensions = 30  # Set dimensions between 10~30
    max_iterations = 1000  # Equivalent to generations in GA
    evaluation_limit = 200000  # Total evaluations limit

    # Unimodal Schwefel P2.22
    best_position_uni, best_fitness_uni, fitness_history_unimodal, evals_uni = nelder_mead(
        schwefel_unimodal, num_dimensions, -10, 10, max_iterations, evaluation_limit)

    # Multimodal Schwefel
    best_position_multi, best_fitness_multi, fitness_history_multimodal, evals_multi = nelder_mead(
        schwefel_multimodal, num_dimensions, -500, 500, max_iterations, evaluation_limit)

    print(f"Total evaluations for Unimodal: {evals_uni}")
    print(f"Best Fitness for Unimodal: {best_fitness_uni}")
    print(f"Total evaluations for Multimodal: {evals_multi}")
    print(f"Best Fitness for Multimodal: {best_fitness_multi}")

    # Plot the comparison
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history_unimodal, label="Unimodal (Schwefel P2.22)", color="blue")
    plt.plot(fitness_history_multimodal, label="Multimodal (Schwefel)", color="red")
    plt.yscale('log')  # Optional: Logarithmic scale for y-axis
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness (log scale)")
    plt.title("NM Performance Comparison on Unimodal vs. Multimodal Functions with 200,000 Evaluations")
    plt.legend()
    plt.grid(True)
    plt.show()
