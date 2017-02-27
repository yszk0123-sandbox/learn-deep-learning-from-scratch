import numpy as np


class AdaGrad:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            for key, value in params.items():
                self.h[key] = np.zeros_like(value)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= \
                self.learning_rate * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
