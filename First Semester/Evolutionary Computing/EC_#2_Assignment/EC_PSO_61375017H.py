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

# Particle Swarm Optimization
class Particle:
    def __init__(self, dim, lower_bound, upper_bound):
        self.position = np.array([random.uniform(lower_bound, upper_bound) for _ in range(dim)])
        self.velocity = np.array([random.uniform(-1, 1) for _ in range(dim)])
        self.best_position = np.copy(self.position)    # personal best position
        self.best_fitness = float('inf')    # best fitness achieved

    # Updates the particle’s velocity using the PSO formula based on inertia, cognitive, and social factors.
    def update_velocity(self, global_best_position, inertia, cognitive_coeff, social_coeff):
        cognitive = cognitive_coeff * random.random() * (self.best_position - self.position)
        social = social_coeff * random.random() * (global_best_position - self.position)
        self.velocity = inertia * self.velocity + cognitive + social

    # Updates the particle’s position by adding the velocity and ensures it stays within bounds.
    def update_position(self, lower_bound, upper_bound):
        self.position += self.velocity
        # Apply bounds to ensure particles stay within limits
        self.position = np.clip(self.position, lower_bound, upper_bound)

def pso(fitness_function, dim, lower_bound, upper_bound, num_particles, max_iterations, evaluation_limit=200000):
    # PSO Parameters
    # For inertia, higher values encourage exploration, while lower values promote convergence.
    inertia = 0.7         # Inertia weight
    # For Cognitive and Social Coefficient, higher values increase responsiveness to personal and global bests.
    cognitive_coeff = 1.5 # Cognitive coefficient
    social_coeff = 1.5    # Social coefficient

    # Initialize particles
    swarm = [Particle(dim, lower_bound, upper_bound) for _ in range(num_particles)]
    global_best_position = np.zeros(dim)
    global_best_fitness = float('inf')

    fitness_history = []
    total_evaluations = 0

    # PSO Main Loop
    for iteration in range(max_iterations):
        for particle in swarm:
            # Evaluate particle fitness
            fitness = fitness_function(particle.position)
            total_evaluations += 1
            
            # Update personal best
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = np.copy(particle.position)

            # Update global best
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = np.copy(particle.position)

            # Check evaluation limit
            if total_evaluations >= evaluation_limit:
                break

        # Record the best fitness for the current iteration
        fitness_history.append(global_best_fitness)

        # Update velocity and position of each particle
        for particle in swarm:
            particle.update_velocity(global_best_position, inertia, cognitive_coeff, social_coeff)
            particle.update_position(lower_bound, upper_bound)

        # Check evaluation limit again
        if total_evaluations >= evaluation_limit:
            break

    return global_best_position, global_best_fitness, fitness_history, total_evaluations

if __name__ == "__main__":
    num_dimensions = 30  # Set dimensions between 10~30
    num_particles = 200  # Equivalent to population size in GA
    max_iterations = 1000  # Equivalent to generations in GA
    evaluation_limit = 200000  # Total evaluations limit

    # Unimodal Schwefel P2.22
    best_position_uni, best_fitness_uni, fitness_history_unimodal, evals_uni = pso(
        schwefel_unimodal, num_dimensions, -10, 10, num_particles, max_iterations, evaluation_limit)

    # Multimodal Schwefel
    best_position_multi, best_fitness_multi, fitness_history_multimodal, evals_multi = pso(
        schwefel_multimodal, num_dimensions, -500, 500, num_particles, max_iterations, evaluation_limit)

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
    plt.title("PSO Performance Comparison on Unimodal vs. Multimodal Functions with 200,000 Evaluations")
    plt.legend()
    plt.grid(True)
    plt.show()