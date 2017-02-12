import numpy as np


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        old_xi = x[i]

        x[i] = old_xi + h
        y1 = f(x)

        x[i] = old_xi - h
        y2 = f(x)

        grad[i] = (y1 - y2) / (2 * h)
        x[i] = old_xi

    return grad


def f(x):
    return np.sum(x * x)


if __name__ == '__main__':
    print(numerical_gradient(f, np.array([3.0, 4.0])))
    print(numerical_gradient(f, np.array([0.0, 2.0])))
    print(numerical_gradient(f, np.array([3.0, 0.0])))
