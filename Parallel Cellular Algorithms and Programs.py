import numpy as np

# Step 1: Define the Rastrigin function
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# Step 2: Initialize parameters
grid_size = (10, 10)  # 10x10 grid of cells
num_cells = grid_size[0] * grid_size[1]  # Total number of cells
mutation_rate = 0.1  # Mutation rate
num_iterations = 100  # Number of iterations
solution_space_bounds = (-5.12, 5.12)  # Rastrigin function bounds

# Initialize the population of cells with random positions in the solution space
def initialize_population(grid_size):
    return np.random.uniform(solution_space_bounds[0], solution_space_bounds[1],
                             (grid_size[0], grid_size[1], 2))  # 2 for x and y dimensions

# Step 3: Evaluate the fitness of each cell
def evaluate_fitness(grid):
    fitness = np.zeros((grid_size[0], grid_size[1]))
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            fitness[i, j] = rastrigin(grid[i, j])  # Apply the rastrigin function to each cell's position
    return fitness

# Step 4: Update states (cells move toward better neighbors)
def update_states(grid, fitness):
    new_grid = grid.copy()
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Get the current cell's neighbors (Von Neumann neighborhood)
            neighbors = get_neighbors(i, j)
            best_neighbor = min(neighbors, key=lambda n: fitness[n[0], n[1]])
            new_grid[i, j] = grid[best_neighbor[0], best_neighbor[1]]  # Move toward best neighbor
            
            # Optionally apply mutation (small random change)
            if np.random.rand() < mutation_rate:
                new_grid[i, j] += np.random.uniform(-0.1, 0.1, size=2)  # Mutate with small random values
                new_grid[i, j] = np.clip(new_grid[i, j], solution_space_bounds[0], solution_space_bounds[1])  # Keep within bounds
    return new_grid

# Get the Von Neumann neighbors of a cell (top, bottom, left, right)
def get_neighbors(i, j):
    neighbors = []
    if i > 0: neighbors.append((i-1, j))  # Top
    if i < grid_size[0]-1: neighbors.append((i+1, j))  # Bottom
    if j > 0: neighbors.append((i, j-1))  # Left
    if j < grid_size[1]-1: neighbors.append((i, j+1))  # Right
    return neighbors

# Step 5: Run the PCA algorithm
def parallel_cellular_algorithm():
    grid = initialize_population(grid_size)
    best_solution = None
    best_fitness = float('inf')

    for iteration in range(num_iterations):
        fitness = evaluate_fitness(grid)
        
        # Track the best solution
        min_fitness_idx = np.unravel_index(np.argmin(fitness), fitness.shape)
        if fitness[min_fitness_idx] < best_fitness:
            best_fitness = fitness[min_fitness_idx]
            best_solution = grid[min_fitness_idx]

        # Update the grid based on the fitness of its neighbors
        grid = update_states(grid, fitness)
        
        print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

    return best_solution, best_fitness

# Step 6: Output the best solution
best_solution, best_fitness = parallel_cellular_algorithm()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness (Rastrigin): {best_fitness}")
