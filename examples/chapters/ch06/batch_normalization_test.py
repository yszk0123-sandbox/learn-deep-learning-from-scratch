import numpy as np
import matplotlib.pyplot as plt
from examples.dataset.mnist import load_mnist
from examples.common.multi_layer_net_extend import MultiLayerNetExtend
from examples.common.optimizer import SGD, Adam

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01


def train(weight_init_std):
    batch_normalization_network = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100],
        output_size=10)
    network = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100],
        output_size=10,
        weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)

    train_accuracy_list = []
    batch_normalization_train_accuracy_list = []

    iteration_per_epoch = max(train_size / batch_size, 1)
    epoch_count = 0

    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (batch_normalization_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        if i % iteration_per_epoch == 0:
            train_accuracy = network.accuracy(x_train, t_train)
            batch_normalization_train_accuracy = \
                batch_normalization_network.accuracy(x_train, t_train)
            train_accuracy_list.append(train_accuracy)
            batch_normalization_train_accuracy_list \
                .append(batch_normalization_train_accuracy)

            print("epoch:" + str(epoch_count) + " | " +
                  str(train_accuracy) +
                  " - " +
                  str(batch_normalization_train_accuracy))

            epoch_count += 1
            if epoch_count >= max_epochs:
                break

    return train_accuracy_list, batch_normalization_train_accuracy_list

weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)

# TODO
