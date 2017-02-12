import numpy as np


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    delta = 1e-7
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size


def cross_entropy_error_naive(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


y_data = [
    [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],  # good
    [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0],  # bad
]

t_data = [
    [0,   0,    1,   0,   0,    0,   0,   0,   0,   0],
    [0,   0,    1,   0,   0,    0,   0,   0,   0,   0],
]

for (y, t) in zip(y_data, t_data):
    print(cross_entropy_error(np.array(y), np.array(t)))
