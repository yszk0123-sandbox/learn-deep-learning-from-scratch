import numpy as np
import matplotlib.pylab as plt
from examples.dataset.mnist import load_mnist
from .two_layer_neural_network import TwoLayerNeuralNetwork

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_accuracy_list = []
test_accuracy_list = []

# Hyperparameter
iteration_count = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

iteration_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNeuralNetwork(input_size=784, hidden_size=50, output_size=10)

for i in range(iteration_count):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iteration_per_epoch == 0:
        train_accuracy = network.accuracy(x_train, t_train)
        test_accuracy = network.accuracy(x_test, t_test)
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)
        print("accuracy: train({train}), test({test})"
              .format(train=train_accuracy, test=test_accuracy))


x = range(iteration_count)
y = train_loss_list
plt.xlabel('iteration')
plt.ylabel('loss')
plt.plot(x, y)
plt.show()
