import numpy as np
import random
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

# Particle Swarm Optimization (Global Search)
class Particle:
    def __init__(self, dim, lower_bound, upper_bound):
        self.position = np.array([random.uniform(lower_bound, upper_bound) for _ in range(dim)])
        self.velocity = np.array([random.uniform(-1, 1) for _ in range(dim)])
        self.best_position = np.copy(self.position)    # personal best position
        self.best_fitness = float('inf')    # best fitness achieved

    def update_velocity(self, global_best_position, inertia, cognitive_coeff, social_coeff):
        cognitive = cognitive_coeff * random.random() * (self.best_position - self.position)
        social = social_coeff * random.random() * (global_best_position - self.position)
        self.velocity = inertia * self.velocity + cognitive + social

    def update_position(self, lower_bound, upper_bound):
        self.position += self.velocity
        self.position = np.clip(self.position, lower_bound, upper_bound)

def pso(fitness_function, dim, lower_bound, upper_bound, num_particles, max_iterations, evaluation_limit=200000):
    inertia = 0.7
    cognitive_coeff = 1.5
    social_coeff = 1.5

    swarm = [Particle(dim, lower_bound, upper_bound) for _ in range(num_particles)]
    global_best_position = np.zeros(dim)
    global_best_fitness = float('inf')

    fitness_history = []
    total_evaluations = 0

    for iteration in range(max_iterations):
        for particle in swarm:
            fitness = fitness_function(particle.position)
            total_evaluations += 1

            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = np.copy(particle.position)

            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = np.copy(particle.position)

            if total_evaluations >= evaluation_limit:
                break

        fitness_history.append(global_best_fitness)

        for particle in swarm:
            particle.update_velocity(global_best_position, inertia, cognitive_coeff, social_coeff)
            particle.update_position(lower_bound, upper_bound)

        if total_evaluations >= evaluation_limit:
            break

    return global_best_position, global_best_fitness, fitness_history, total_evaluations

# Nelder-Mead (Local Refinement)
def nelder_mead(fitness_function, initial_guess, lower_bound, upper_bound, max_iterations, evaluation_limit=200000):
    fitness_history = []
    total_evaluations = [0]

    def callback(xk):
        fitness_history.append(fitness_function(xk))

    def fitness_with_count(x):
        if total_evaluations[0] >= evaluation_limit:
            return float('inf')
        total_evaluations[0] += 1
        return fitness_function(x)

    result = minimize(
        fitness_with_count,
        initial_guess,
        method='Nelder-Mead',
        options={'maxiter': max_iterations, 'adaptive': True},
        callback=callback
    )

    return result.x, result.fun, fitness_history, total_evaluations[0]

# Hybrid PSO-NM
def pso_nm_hybrid(fitness_function, dim, lower_bound, upper_bound, num_particles, pso_iterations, nm_iterations, evaluation_limit):
    # Step 1: Global Search with PSO
    best_position_pso, best_fitness_pso, fitness_history_pso, evals_pso = pso(
        fitness_function, dim, lower_bound, upper_bound, num_particles, pso_iterations, evaluation_limit
    )

    remaining_evaluations = evaluation_limit - evals_pso
    if remaining_evaluations <= 0:
        return best_position_pso, best_fitness_pso, fitness_history_pso, evals_pso

    # Step 2: Local Refinement with NM
    best_position_nm, best_fitness_nm, fitness_history_nm, evals_nm = nelder_mead(
        fitness_function, best_position_pso, lower_bound, upper_bound, nm_iterations, remaining_evaluations
    )

    # Combine fitness histories for plotting
    fitness_history = fitness_history_pso + fitness_history_nm
    total_evaluations = evals_pso + evals_nm

    return best_position_nm, best_fitness_nm, fitness_history, total_evaluations

if __name__ == "__main__":
    num_dimensions = 30
    num_particles = 200
    pso_iterations = 500
    nm_iterations = 500
    evaluation_limit = 200000

    # Unimodal Schwefel P2.22
    best_position_uni, best_fitness_uni, fitness_history_unimodal, evals_uni = pso_nm_hybrid(
        schwefel_unimodal, num_dimensions, -10, 10, num_particles, pso_iterations, nm_iterations, evaluation_limit
    )

    # Multimodal Schwefel
    best_position_multi, best_fitness_multi, fitness_history_multimodal, evals_multi = pso_nm_hybrid(
        schwefel_multimodal, num_dimensions, -500, 500, num_particles, pso_iterations, nm_iterations, evaluation_limit
    )

    print(f"Total evaluations for Unimodal: {evals_uni}")
    print(f"Best Fitness for Unimodal: {best_fitness_uni}")
    print(f"Total evaluations for Multimodal: {evals_multi}")
    print(f"Best Fitness for Multimodal: {best_fitness_multi}")

    # Plot the comparison
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history_unimodal, label="Unimodal (Schwefel P2.22)", color="blue")
    plt.plot(fitness_history_multimodal, label="Multimodal (Schwefel)", color="red")
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness (log scale)")
    plt.title("PSO-NM Hybrid Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()
