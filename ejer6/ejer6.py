import numpy as np
import random

# Configuración del problema
graph = np.array([
    [0, 7, 9, 10, 20],
    [7, 0, np.inf, 8, 4],
    [9, np.inf, 0, 15, 5],
    [10, 8, 15, 0, 11],
    [20, 4, 5, 11, 0]
])

num_nodes = len(graph)
population_size = 10
num_generations = 100
mutation_rate = 0.1

# Función para calcular la distancia total de una ruta
def calculate_distance(route):
    distance = 0
    for i in range(len(route) - 1):
        distance += graph[route[i], route[i + 1]]
    distance += graph[route[-1], route[0]]  # Retorno al nodo inicial
    return distance

# Función para crear una ruta aleatoria
def create_route():
    route = list(range(num_nodes))
    random.shuffle(route)
    return route

# Función para crear la población inicial
def create_initial_population():
    return [create_route() for _ in range(population_size)]

# Función para seleccionar dos padres usando el método de torneo
def select_parents(population):
    tournament_size = 3
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=calculate_distance)
    return tournament[0], tournament[1]

# Función para cruzar dos padres y generar descendencia
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(num_nodes), 2))
    child = [None] * num_nodes
    child[start:end + 1] = parent1[start:end + 1]
    pointer = 0
    for node in parent2:
        if node not in child:
            while child[pointer] is not None:
                pointer += 1
            child[pointer] = node
    return child

# Función para mutar una ruta
def mutate(route):
    if random.random() < mutation_rate:
        i, j = random.sample(range(num_nodes), 2)
        route[i], route[j] = route[j], route[i]

# Función principal del algoritmo genético
def genetic_algorithm():
    population = create_initial_population()
    for generation in range(num_generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            mutate(child1)
            mutate(child2)
            new_population.append(child1)
            new_population.append(child2)
        population = new_population
        best_route = min(population, key=calculate_distance)
        print(f'Generación {generation + 1}: Distancia mínima = {calculate_distance(best_route)}')
    return best_route

# Ejecutar el algoritmo genético
best_route = genetic_algorithm()
print(f'Mejor ruta: {best_route}')
print(f'Distancia de la mejor ruta: {calculate_distance(best_route)}')
