import numpy as np
import random
import matplotlib.pyplot as plt

# Schwefel's P2.22 Function (Unimodal)
def schwefel_unimodal(x):
    schwefel_uni = np.sum(np.abs(x)) + np.prod(np.abs(x), axis=0)
    return schwefel_uni

# Schwefel's Function (Multimodal)
def schwefel_multimodal(x):
    d = len(x)
    schwefel_muti = 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=0)
    return schwefel_muti

# Simulated Annealing Algorithm
def simulated_annealing(fitness_function, dim, lower_bound, upper_bound, max_iterations, evaluation_limit=200000):
    # Initial parameters
    initial_temp = 1000  # Starting temperature
    cooling_rate = 0.995  # Cooling factor (closer to 1 means slower cooling)
    temperature = initial_temp

    # Initialize the current solution
    current_position = np.random.uniform(lower_bound, upper_bound, dim)
    current_fitness = fitness_function(current_position)

    best_position = np.copy(current_position)
    best_fitness = current_fitness

    fitness_history = []
    total_evaluations = 1

    # SA Main Loop
    for iteration in range(max_iterations):
        # Generate a neighbor solution
        neighbor_position = current_position + np.random.uniform(-1, 1, dim)
        # Clip the neighbor to ensure it stays within bounds
        neighbor_position = np.clip(neighbor_position, lower_bound, upper_bound)

        # Evaluate the neighbor's fitness
        neighbor_fitness = fitness_function(neighbor_position)
        total_evaluations += 1

        # Acceptance Criteria: Accept if better, or probabilistically if worse
        if (neighbor_fitness < current_fitness) or (
            random.random() < np.exp((current_fitness - neighbor_fitness) / temperature)
        ):
            current_position = neighbor_position
            current_fitness = neighbor_fitness

        # Update the best solution found so far
        if current_fitness < best_fitness:
            best_position = np.copy(current_position)
            best_fitness = current_fitness

        # Record the best fitness for plotting
        fitness_history.append(best_fitness)

        # Decrease the temperature
        temperature *= cooling_rate

        # Stop if evaluation limit is reached
        if total_evaluations >= evaluation_limit:
            break

    return best_position, best_fitness, fitness_history, total_evaluations

if __name__ == "__main__":
    num_dimensions = 30  # Set dimensions between 10~30
    max_iterations = 1000  # Equivalent to generations in GA
    evaluation_limit = 200000  # Total evaluations limit

    # Unimodal Schwefel P2.22
    best_position_uni, best_fitness_uni, fitness_history_unimodal, evals_uni = simulated_annealing(
        schwefel_unimodal, num_dimensions, -10, 10, max_iterations, evaluation_limit
    )

    # Multimodal Schwefel
    best_position_multi, best_fitness_multi, fitness_history_multimodal, evals_multi = simulated_annealing(
        schwefel_multimodal, num_dimensions, -500, 500, max_iterations, evaluation_limit
    )

    print(f"Total evaluations for Unimodal: {evals_uni}")
    print(f"Best Fitness for Unimodal: {best_fitness_uni}")
    print(f"Total evaluations for Multimodal: {evals_multi}")
    print(f"Best Fitness for Multimodal: {best_fitness_multi}")

    # Plot the comparison
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history_unimodal, label="Unimodal (Schwefel P2.22)", color="blue")
    plt.plot(fitness_history_multimodal, label="Multimodal (Schwefel)", color="red")
    plt.yscale("log")  # Optional: Logarithmic scale for y-axis
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness (log scale)")
    plt.title("SA Performance Comparison on Unimodal vs. Multimodal Functions with 200,000 Evaluations")
    plt.legend()
    plt.grid(True)
    plt.show()
