import numpy as np
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

# Snake-Tongue Algorithm
def snake_tongue_algorithm(fitness_function, dim, lower_bound, upper_bound, max_iterations, evaluation_limit=200000):
    # Initialize random starting position within bounds
    position = np.random.uniform(lower_bound, upper_bound, dim)
    fitness = fitness_function(position)
    total_evaluations = 1

    # Initialize history tracking
    fitness_history = [fitness]

    # Set initial exploration radius and decay rate
    exploration_radius = (upper_bound - lower_bound) * 0.1
    decay_rate = 0.99  # Reduce exploration radius over time

    for iteration in range(max_iterations):
        if total_evaluations >= evaluation_limit:
            break

        # Generate new candidates around the current position
        candidates = [
            position + np.random.uniform(-exploration_radius, exploration_radius, dim)
            for _ in range(10)
        ]

        # Clip candidates to remain within bounds
        candidates = np.clip(candidates, lower_bound, upper_bound)

        # Evaluate all candidates and find the best one
        candidate_fitness = [fitness_function(candidate) for candidate in candidates]
        total_evaluations += len(candidates)

        # Update position and fitness if a better candidate is found
        best_candidate_idx = np.argmin(candidate_fitness)
        if candidate_fitness[best_candidate_idx] < fitness:
            fitness = candidate_fitness[best_candidate_idx]
            position = candidates[best_candidate_idx]

        # Record the best fitness for this iteration
        fitness_history.append(fitness)

        # Decay exploration radius to balance exploration and exploitation
        exploration_radius *= decay_rate

        # Check evaluation limit
        if total_evaluations >= evaluation_limit:
            break

    return position, fitness, fitness_history, total_evaluations

if __name__ == "__main__":
    num_dimensions = 30  # Set dimensions between 10~30
    max_iterations = 1000  # Maximum iterations
    evaluation_limit = 200000  # Total evaluations limit

    # Unimodal Schwefel P2.22
    best_position_uni, best_fitness_uni, fitness_history_unimodal, evals_uni = snake_tongue_algorithm(
        schwefel_unimodal, num_dimensions, -10, 10, max_iterations, evaluation_limit
    )

    # Multimodal Schwefel
    best_position_multi, best_fitness_multi, fitness_history_multimodal, evals_multi = snake_tongue_algorithm(
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
    plt.yscale('log')  # Optional: Logarithmic scale for y-axis
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness (log scale)")
    plt.title("Snake-Tongue Algorithm Performance on Unimodal vs. Multimodal Functions")
    plt.legend()
    plt.grid(True)
    plt.show()
