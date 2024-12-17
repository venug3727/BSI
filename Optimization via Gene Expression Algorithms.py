import numpy as np

# Step 1: Define a more complex problem - Rastrigin function
def objective_function(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# Step 2: Initialize parameters
population_size = 50
num_genes = 10  # Can adjust the number of dimensions (features)
mutation_rate = 0.01
crossover_rate = 0.8
num_generations = 100

# Step 3: Initialize population (random values in range [-5.12, 5.12] for Rastrigin)
population = np.random.uniform(-5.12, 5.12, (population_size, num_genes))

# Step 4: Evaluate fitness
def evaluate_fitness(population):
    fitness = []
    for individual in population:
        fitness.append(objective_function(individual))
    return np.array(fitness)

# Step 5: Selection (roulette wheel selection)
def select_population(population, fitness):
    selected = []
    fitness_inv = 1 / (1 + fitness)  # Inverse of fitness (lower fitness should have less chance)
    prob = fitness_inv / fitness_inv.sum()  # Probability based on fitness
    indices = np.random.choice(np.arange(len(population)), size=len(population), p=prob)
    selected = population[indices]
    return selected

# Step 6: Crossover
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1)-1)
    offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return offspring1, offspring2

# Step 7: Mutation
def mutate(offspring):
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            offspring[i] = np.random.uniform(-5.12, 5.12)  # Random mutation within the bounds
    return offspring

# Step 8: Gene Expression (mapping genetic sequences to solutions)
def gene_expression(binary):
    return binary  # For this case, the solution is directly the real-valued solution

# Main loop
best_solution = None
best_fitness = float('inf')
for generation in range(num_generations):
    fitness = evaluate_fitness(population)
    selected = select_population(population, fitness)
    
    # Crossover and mutation
    offspring_population = []
    for i in range(0, len(selected), 2):
        parent1, parent2 = selected[i], selected[i+1]
        if np.random.rand() < crossover_rate:
            offspring1, offspring2 = crossover(parent1, parent2)
        else:
            offspring1, offspring2 = parent1, parent2
        offspring_population.append(mutate(offspring1))
        offspring_population.append(mutate(offspring2))
    
    population = np.array(offspring_population)
    
    # Track best solution
    min_fitness_idx = np.argmin(fitness)
    if fitness[min_fitness_idx] < best_fitness:
        best_fitness = fitness[min_fitness_idx]
        best_solution = population[min_fitness_idx]
    
    print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

# Final solution
print(f"Best Solution: {best_solution}")
print(f"Best Fitness (Rastrigin): {best_fitness}")
