import numpy as np
from examples.dataset.mnist import load_mnist
from .two_layer_neural_network import TwoLayerNeuralNetwork

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNeuralNetwork(input_size=784, hidden_size=50, output_size=10)
network = TwoLayerNeuralNetwork(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backword_propagation = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(
        grad_backword_propagation[key] - grad_numerical[key]
    ))
    print(key + ":" + str(diff))
