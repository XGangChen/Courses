import numpy as np
import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the problem as minimization
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Schwefel's P2.22 Function
def schwefel_unimodal(x):
    schwefel_uni = np.sum(np.abs(x)) + np.prod(np.abs(x), axis=0)
    return schwefel_uni

# Schwefel's Function
def schwefel_multimodal(x):
    d = len(x)
    schwefel_muti = 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=0)
    return schwefel_muti

# Function to create an individual with random values
def create_individual(n, lower_bound, upper_bound):
    id = [random.uniform(lower_bound, upper_bound) for _ in range(n)]
    return id

# Evaluation function for unimodal Schwefel's P2.22
def eval_schwefel_unimodal(individual):
    return schwefel_unimodal(individual),

# Evaluation function for multimodal Schwefel function
def eval_schwefel_multimodal(individual):
    return schwefel_multimodal(individual),

# Setup GA parameters
def setup_ga(evaluate_function, num_dimensions, lower_bound, upper_bound):
    toolbox = base.Toolbox()
    
    # Define how to generate individuals and population
    toolbox.register("attr_float", random.uniform, lower_bound, upper_bound)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, num_dimensions)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register the evaluation, selection, crossover, and mutation functions
    toolbox.register("evaluate", evaluate_function)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)

# --------------------Tournament selection
    toolbox.register("select", tools.selTournament, tournsize=3)
# --------------------Rank-based selection(pure ranking)
    # toolbox.register("select", tools.selBest)
# --------------------Roulette Wheel Selection (proportional to rank)
    # toolbox.register("select", tools.selRoulette)
    
    # Register the algorithm to use a simple evolutionary algorithm
    toolbox.register("map", map)
    
    return toolbox

# Modify the GA function to stop at 200,000 function evaluations
def run_ga_with_evaluation_limit(toolbox, population_size, num_generations, evaluation_limit=200000):
    pop = toolbox.population(n=population_size)
    fitness_history = []
    total_evaluations = 0
    
    for gen in range(num_generations):
        # Evaluate individuals
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.5)
        fits = list(map(toolbox.evaluate, offspring))

        # Update the population with evaluated offspring
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
        pop = toolbox.select(offspring, k=len(pop))

        # Track the best fitness at each generation
        best_ind = tools.selBest(pop, k=1)[0]
        fitness_history.append(best_ind.fitness.values[0])

        # Update the total function evaluations
        total_evaluations += len(fits)
        
        # Stop if we reach the evaluation limit
        if total_evaluations >= evaluation_limit:
            break
    
    return best_ind, fitness_history, total_evaluations

if __name__ == "__main__":
    num_dimensions = 30    # Set dimensions between 10~30
    population_size = 200  # Example population size
    num_generations = 1000  # Example number of generations
    # Either population_size or num_generations can adjusts, but the num of "population_size * num_generations" must be 200,000.
    evaluation_limit = 200000  # Set the total function evaluations limit
    
    # Unimodal Schwefel P2.22
    toolbox_unimodal = setup_ga(eval_schwefel_unimodal, num_dimensions, -10, 10)
    _, fitness_history_unimodal, evals_uni = run_ga_with_evaluation_limit(toolbox_unimodal, population_size, num_generations, evaluation_limit)
    
    # Multimodal Schwefel
    toolbox_multimodal = setup_ga(eval_schwefel_multimodal, num_dimensions, -500, 500)
    _, fitness_history_multimodal, evals_multi = run_ga_with_evaluation_limit(toolbox_multimodal, population_size, num_generations, evaluation_limit)
    
    print(f"Total evaluations for Unimodal: {evals_uni}")
    print(f"Total evaluations for Multimodal: {evals_multi}")
    
    # Plot the comparison
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history_unimodal, label="Unimodal (Schwefel P2.22)", color="blue")
    plt.plot(fitness_history_multimodal, label="Multimodal (Schwefel)", color="red")
    plt.yscale('log')  # Optional: Logarithmic scale for y-axis
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (log scale)")
    plt.title("GA Performance Comparison on Unimodal vs. Multimodal Functions with 200,000 Evaluations")
    plt.legend()
    plt.grid(True)
    plt.show()

    # # Create a grid of values for x1 and x2
    # x1 = np.linspace(-500, 500, 100)
    # x2 = np.linspace(-500, 500, 100)
    # x1, x2 = np.meshgrid(x1, x2)

    # # Combine x1 and x2 into a single array
    # X = np.array([x1, x2])

    # # Apply the Schwefel function to the grid
    # Z_uni = schwefel_unimodal(X)
    # Z_multi = schwefel_multimodal(X)

    # # Create a 3D plot of umimodal
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # # Plot the unimodal surface
    # ax.plot_surface(x1, x2, Z_uni, cmap='viridis')
    # # Add labels
    # ax.set_xlabel('x1')
    # ax.set_ylabel('x2')
    # ax.set_zlabel('f(x1, x2)')
    # ax.set_title('Multimodal Schwefel Function')
    # # Show the plot
    # plt.show()

    # # Create a 3D plot of multimodal
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # # Plot the multimodal surface
    # ax.plot_surface(x1, x2, Z_multi, cmap='viridis')
    # # Add labels
    # ax.set_xlabel('x1')
    # ax.set_ylabel('x2')
    # ax.set_zlabel('f(x1, x2)')
    # ax.set_title('Multimodal Schwefel Function')
    # # Show the plot
    # plt.show()