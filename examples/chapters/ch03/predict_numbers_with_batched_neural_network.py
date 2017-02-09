from examples.dataset.mnist import load_mnist
import pickle
import numpy as np
from . import activation_functions


def get_data():
    (x_train, t_train), (x_test, t_test) = \
            load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("examples/dataset/chapters/ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = activation_functions.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = activation_functions.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = activation_functions.softmax(a3)

    return y


xs, ts = get_data()
network = init_network()

batch_size = 100
accuracy_count = 0
xs_length = len(xs)

for i in range(0, xs_length, batch_size):
    y_batch = predict(network, xs[i:i+batch_size])
    p = np.argmax(y_batch, axis=1)
    accuracy_count += np.sum(p == ts[i:i+batch_size])

    print("Accuracy: " + str(float(accuracy_count) / xs_length))
