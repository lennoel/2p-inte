import numpy as np
import matplotlib.pyplot as plt

# Generar un conjunto de ciudades y una matriz de distancias aleatorias usando enteros
np.random.seed(0)
num_cities = 10
cities = np.random.randint(0, 100, size=(num_cities, 2))  # Coordenadas (x, y) de las ciudades

# Calcular la matriz de distancias usando enteros
dist_matrix = np.round(np.sqrt(((cities[:, np.newaxis, :] - cities[np.newaxis, :, :]) ** 2).sum(axis=2))).astype(int)

# Mostrar la matriz de distancias
print("Matriz de distancias:")
print(dist_matrix)


# Algoritmo de Vecino Más Cercano
def nearest_neighbor(dist_matrix):
    num_cities = dist_matrix.shape[0]
    visited = [False] * num_cities
    path = [0]  # Empezar desde la ciudad 0
    visited[0] = True

    while len(path) < num_cities:
        last = path[-1]
        nearest = np.argmin([dist_matrix[last, j] if not visited[j] else np.inf for j in range(num_cities)])
        path.append(nearest)
        visited[nearest] = True

    path.append(0)  # Regresar a la ciudad inicial
    return path


# Obtener la ruta usando el algoritmo de vecino más cercano
path = nearest_neighbor(dist_matrix)

# Calcular la longitud total de la ruta
total_distance = sum(dist_matrix[path[i], path[i + 1]] for i in range(num_cities))

# Imprimir la ruta y la distancia total
print("Ruta:", path)
print("Distancia total:", total_distance)

# Graficar la ruta
plt.figure(figsize=(10, 6))
for i in range(num_cities):
    plt.plot(cities[path[i], 0], cities[path[i], 1], 'ro')
    plt.text(cities[path[i], 0], cities[path[i], 1], str(i), fontsize=12, color='blue')

for i in range(num_cities):
    plt.plot([cities[path[i], 0], cities[path[i + 1], 0]], [cities[path[i], 1], cities[path[i + 1], 1]], 'b-')

plt.title('Vecino Más Cercano')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.grid(True)
plt.show()
