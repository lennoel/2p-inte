import numpy as np
import pandas as pd

# Cargar el conjunto de datos
data = pd.read_csv("C:\\Users\\Grinshow\\Documents\\examen2\\Iris.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalizar los datos
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Codificar las etiquetas de clase como vectores one-hot
y_encoded = pd.get_dummies(y).values

# Inicializar los pesos y sesgos
weights = np.random.randn(X.shape[1], y_encoded.shape[1])
biases = np.random.randn(y_encoded.shape[1])

# Definir la tasa de aprendizaje
learning_rate = 0.4

# Función de activación y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Entrenar la red
for epoch in range(10000):  # Número de épocas
    # Forward propagation
    z = np.dot(X, weights) + biases
    a = sigmoid(z)

    # Backward propagation
    dz = a - y_encoded
    dw = np.dot(X.T, dz) / X.shape[0]
    db = np.sum(dz, axis=0) / X.shape[0]

    # Actualizar los pesos y sesgos
    weights -= learning_rate * dw
    biases -= learning_rate * db

    # Imprimir el error cada 1000 épocas
    y_encoded = pd.get_dummies(y).values.astype(int)
    if epoch % 1000 == 0:
        loss = np.mean(-y_encoded * np.log(a) - (1 - y_encoded) * np.log(1 - a))
        print(f"Epoca {epoch}, perdida: {loss}")
