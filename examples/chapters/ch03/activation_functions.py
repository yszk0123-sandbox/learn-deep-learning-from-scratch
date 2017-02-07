import numpy as np


def identity(x):
    return x


def step(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax_naive(a):
    exp_a = np.exp(a)
    return exp_a / np.sum(exp_a)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a)
