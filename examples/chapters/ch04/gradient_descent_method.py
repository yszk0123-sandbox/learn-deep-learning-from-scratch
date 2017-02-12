import numpy as np
from . import numerical_gradient as grad


def gradient_descent(f, initial_x, learning_rate=0.01, step_number=100):
    x = initial_x

    for i in range(step_number):
        x -= learning_rate * grad.numerical_gradient(f, x)

    return x


def f(x):
    return x[0] ** 2.0 + x[1] ** 2.0


if __name__ == '__main__':
    print(gradient_descent(f, np.array([-3.0, 4.0])))
    print(gradient_descent(f, np.array([-3.0, 4.0]), learning_rate=0.1))
    print(gradient_descent(f, np.array([-3.0, 4.0]), learning_rate=10.0))
    print(gradient_descent(f, np.array([-3.0, 4.0]), learning_rate=1e-10))
