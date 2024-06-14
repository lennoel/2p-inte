import numpy as np
import random

# Definir las ciudades y la matriz de distancias
cities = ['A', 'B', 'C', 'D', 'E']
distances = np.array([
    [0, 7, 9, 10, 20],
    [7, 0, 15, 8, 4],
    [9, 15, 0, 11, 5],
    [10, 8, 11, 0, 17],
    [20, 4, 5, 17, 0]
])

# Definir parámetros del algoritmo genético
population_size = 10
generations = 100
mutation_rate = 0.1
def create_population(cities, population_size):
    population = []
    for _ in range(population_size):
        individual = cities.copy()
        random.shuffle(individual)
        population.append(individual)
    return population

population = create_population(cities, population_size)
def calculate_fitness(individual, distances):
    fitness = 0
    for i in range(len(individual) - 1):
        start = cities.index(individual[i])
        end = cities.index(individual[i + 1])
        fitness += distances[start, end]
    # Añadir la distancia para regresar al punto inicial
    start = cities.index(individual[-1])
    end = cities.index(individual[0])
    fitness += distances[start, end]
    return fitness

def evaluate_population(population, distances):
    fitness_scores = []
    for individual in population:
        fitness = calculate_fitness(individual, distances)
        fitness_scores.append(fitness)
    return fitness_scores

fitness_scores = evaluate_population(population, distances)
def selection(population, fitness_scores):
    selected = np.argsort(fitness_scores)[:len(population)//2]
    return [population[i] for i in selected]

selected_population = selection(population, fitness_scores)


def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end + 1] = parent1[start:end + 1]

    ptr = 0
    for gene in parent2:
        if gene not in child:
            while child[ptr] is not None:
                ptr += 1
            child[ptr] = gene
    return child


def create_new_population(selected_population):
    new_population = selected_population[:]
    while len(new_population) < population_size:
        parent1, parent2 = random.sample(selected_population, 2)
        child = crossover(parent1, parent2)
        new_population.append(child)
    return new_population


new_population = create_new_population(selected_population)
def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def apply_mutation(population, mutation_rate):
    mutated_population = [mutate(individual, mutation_rate) for individual in population]
    return mutated_population

mutated_population = apply_mutation(new_population, mutation_rate)
def genetic_algorithm(cities, distances, population_size, generations, mutation_rate):
    population = create_population(cities, population_size)
    for _ in range(generations):
        fitness_scores = evaluate_population(population, distances)
        selected_population = selection(population, fitness_scores)
        new_population = create_new_population(selected_population)
        population = apply_mutation(new_population, mutation_rate)
    best_index = np.argmin(evaluate_population(population, distances))
    best_route = population[best_index]
    best_distance = calculate_fitness(best_route, distances)
    return best_route, best_distance

best_route, best_distance = genetic_algorithm(cities, distances, population_size, generations, mutation_rate)
print("Mejor ruta:", best_route)
print("Distancia total:", best_distance)
