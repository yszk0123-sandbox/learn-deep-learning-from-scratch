import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def initial_xavier(n):
    return np.random.randn(n, n) / np.sqrt(n)


def initial_he(n):
    return 2 * np.random.randn(n, n) / np.sqrt(n)


def draw_activations_histogram(activations):
    for i, activation in activations.items():
        plt.subplot(1, len(activations), i + 1)
        plt.title(str(i + 1) + "-layer")
        plt.hist(activation.flatten(), 30, range=(0, 1))
    plt.show()


input_data = np.random.randn(1000, 100)
node_count = 100
hidden_layer_size = 5
activations = {}

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]

    w = initial_xavier(node_count)
    # w = initial_he(node_count)

    a = np.dot(x, w)
    z = sigmoid(a)
    # z = np.tanh(a)
    # z = relu(a)
    activations[i] = z

draw_activations_histogram(activations)
