import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
import pandas as pd

# Schwefel's P2.22 Function (Unimodal)
def schwefel_unimodal(x):
    schwefel_uni = np.sum(np.abs(x)) + np.prod(np.abs(x), axis=0)
    return schwefel_uni

# Schwefel's Function (Multimodal)
def schwefel_multimodal(x):
    d = len(x)
    schwefel_muti = 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=0)
    return schwefel_muti

### Genetic Algorithm
def genetic_algorithm(fitness_function, dim, lower_bound, upper_bound, population_size, max_generations, evaluation_limit=200000):
    population = [np.random.uniform(lower_bound, upper_bound, dim) for _ in range(population_size)]
    fitness_history = []
    total_evaluations = 0

    for generation in range(max_generations):
        # Evaluate fitness
        fitness = [fitness_function(ind) for ind in population]
        total_evaluations += len(population)

        # Record best fitness
        best_idx = np.argmin(fitness)
        fitness_history.append(fitness[best_idx])

        # Selection (Tournament)
        selected = [population[np.random.randint(0, len(population))] for _ in range(population_size)]
        
        # Crossover
        offspring = []
        for i in range(0, population_size, 2):
            parent1, parent2 = selected[i], selected[(i+1) % population_size]
            point = np.random.randint(1, dim)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            offspring.append(child1)
            offspring.append(child2)

        # Mutation
        for child in offspring:
            if np.random.rand() < 0.1:  # Mutation rate
                index = np.random.randint(0, dim)
                child[index] += np.random.uniform(-1, 1)

        # Update population
        population = [np.clip(ind, lower_bound, upper_bound) for ind in offspring]

        if total_evaluations >= evaluation_limit:
            break

    return population[best_idx], fitness[best_idx], fitness_history, total_evaluations


### Particle Swarm Optimization
class Particle:
    def __init__(self, dim, lower_bound, upper_bound):
        self.position = np.random.uniform(lower_bound, upper_bound, dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = np.copy(self.position)
        self.best_fitness = float('inf')

    def update_velocity(self, global_best_position, inertia, cognitive_coeff, social_coeff):
        cognitive = cognitive_coeff * np.random.rand() * (self.best_position - self.position)
        social = social_coeff * np.random.rand() * (global_best_position - self.position)
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


### Nelder-Mead
def nelder_mead(fitness_function, dim, lower_bound, upper_bound, max_iterations, evaluation_limit=200000):
    x0 = np.random.uniform(lower_bound, upper_bound, dim)
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
        x0, 
        method='Nelder-Mead', 
        options={'maxiter': max_iterations}, 
        callback=callback
    )

    return result.x, result.fun, fitness_history, total_evaluations[0]


### Simulated Annealing
def simulated_annealing(fitness_function, dim, lower_bound, upper_bound, max_iterations, evaluation_limit=200000):
    bounds = [(lower_bound, upper_bound)] * dim
    fitness_history = []
    total_evaluations = [0]

    def fitness_with_count(x):
        if total_evaluations[0] >= evaluation_limit:
            return float('inf')
        total_evaluations[0] += 1
        return fitness_function(x)

    result = dual_annealing(fitness_with_count, bounds, maxiter=max_iterations)
    fitness_history.append(result.fun)  # This should track the progress
    return result.x, result.fun, fitness_history, total_evaluations[0]



### Snake-Tongue Algorithm
def snake_tongue_algorithm(fitness_function, dim, lower_bound, upper_bound, max_iterations, evaluation_limit=200000):
    position = np.random.uniform(lower_bound, upper_bound, dim)
    fitness = fitness_function(position)
    total_evaluations = 1
    fitness_history = [fitness]

    exploration_radius = (upper_bound - lower_bound) * 0.1
    decay_rate = 0.99

    for iteration in range(max_iterations):
        if total_evaluations >= evaluation_limit:
            break

        candidates = [
            position + np.random.uniform(-exploration_radius, exploration_radius, dim)
            for _ in range(10)
        ]
        candidates = np.clip(candidates, lower_bound, upper_bound)

        candidate_fitness = [fitness_function(candidate) for candidate in candidates]
        total_evaluations += len(candidates)

        best_idx = np.argmin(candidate_fitness)
        if candidate_fitness[best_idx] < fitness:
            fitness = candidate_fitness[best_idx]
            position = candidates[best_idx]

        fitness_history.append(fitness)
        exploration_radius *= decay_rate

    return position, fitness, fitness_history, total_evaluations


if __name__ == "__main__":
    num_dimensions = 30
    max_iterations = 1000
    evaluation_limit = 200000
    population_size = 50  # For GA and PSO

    # Unimodal Schwefel P2.22
    methods = {
        "GA": genetic_algorithm,
        "PSO": pso,
        "NM": nelder_mead,
        "STA": snake_tongue_algorithm,
    }

    results = {}
    for name, method in methods.items():
        if name in ["GA", "PSO"]:
            position, fitness, fitness_history, evals = method(
                schwefel_unimodal, num_dimensions, -10, 10, population_size, max_iterations, evaluation_limit
            )
        else:
            position, fitness, fitness_history, evals = method(
                schwefel_unimodal, num_dimensions, -10, 10, max_iterations, evaluation_limit
            )
        results[name] = fitness_history
        print(f"{name}: Best Fitness = {fitness}, Total Evaluations = {evals}")

    # Plot Unimodal
    plt.figure(figsize=(10, 6))
    for name, history in results.items():
        plt.plot(history, label=name)
    plt.yscale('log')
    plt.xlabel("Iterations")
    plt.ylabel("Best Fitness (log scale)")
    plt.title("Comparison of Optimization Methods (Unimodal)")
    plt.legend()
    plt.grid(True)
    plt.show()


    # Repeat for Multimodal Schwefel Function
    results_multimodal = {}
    for name, method in methods.items():
        if name in ["GA", "PSO"]:
            position, fitness, fitness_history, evals = method(
                schwefel_multimodal, num_dimensions, -500, 500, population_size, max_iterations, evaluation_limit
            )
        else:
            position, fitness, fitness_history, evals = method(
                schwefel_multimodal, num_dimensions, -500, 500, max_iterations, evaluation_limit
            )
        results_multimodal[name] = fitness_history
        print(f"{name} (Multimodal): Best Fitness = {fitness}, Total Evaluations = {evals}")

    # Plot Multimodal
    plt.figure(figsize=(10, 6))
    for name, history in results_multimodal.items():
        plt.plot(history, label=name)
    plt.yscale('log')
    plt.xlabel("Iterations")
    plt.ylabel("Best Fitness (log scale)")
    plt.title("Comparison of Optimization Methods (Multimodal)")
    plt.legend()
    plt.grid(True)
    plt.show()
