import pandas as pd
import numpy as np

# Cargar datos desde el archivo CSV
filename = "C:\\Users\\Grinshow\\Documents\\examen2\Iris.csv"
data = pd.read_csv(filename)

# Preparar las características y las etiquetas
X = data.iloc[:, :-1].values  # Todas las columnas menos la última
y = data.iloc[:, -1].values   # Última columna

# Mapear nombres de clase a números
class_mapping = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}
y = np.array([class_mapping[label] for label in y])

# One-hot encoding de las etiquetas
def one_hot_encode(y):
    n_classes = len(np.unique(y))
    encoded = np.zeros((y.size, n_classes))
    encoded[np.arange(y.size), y] = 1
    return encoded

y = one_hot_encode(y)

# Función de activación escalón
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Inicializar los pesos aleatorios
def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.uniform(-1, 1, (hidden_size, output_size))
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = step_function(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = step_function(z2)
    return a1, a2

# Backward propagation y actualización de pesos
def backward_propagation(X, y, a1, a2, W1, b1, W2, b2, learning_rate):
    error_output = y - a2
    delta_output = error_output  # Derivada de la función escalón es 1 para simplificación

    error_hidden = np.dot(delta_output, W2.T)
    delta_hidden = error_hidden  # Derivada de la función escalón es 1 para simplificación

    W2 += learning_rate * np.dot(a1.T, delta_output)
    b2 += learning_rate * np.sum(delta_output, axis=0, keepdims=True)
    W1 += learning_rate * np.dot(X.T, delta_hidden)
    b1 += learning_rate * np.sum(delta_hidden, axis=0)

# Entrenamiento de la red neuronal
def train(X, y, W1, b1, W2, b2, learning_rate, epochs):
    for epoch in range(epochs):
        a1, a2 = forward_propagation(X, W1, b1, W2, b2)
        backward_propagation(X, y, a1, a2, W1, b1, W2, b2, learning_rate)
        if epoch % 100 == 0:
            loss = np.mean(np.abs(y - a2))
            print(f"Epoch {epoch}, Loss: {loss}")

# Inicializar los parámetros de la red
input_size = X.shape[1]
hidden_size = 5
output_size = y.shape[1]
W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)

# Entrenar la red neuronal
learning_rate = 0.2
epochs = 1000
train(X, y, W1, b1, W2, b2, learning_rate, epochs)

# Predicciones
def predict(X, W1, b1, W2, b2):
    _, a2 = forward_propagation(X, W1, b1, W2, b2)
    return np.argmax(a2, axis=1)

# Evaluación
y_pred = predict(X, W1, b1, W2, b2)
y_true = np.argmax(y, axis=1)

accuracy = np.mean(y_pred == y_true)
print(f"Accuracy: {accuracy * 100:.2f}%")
