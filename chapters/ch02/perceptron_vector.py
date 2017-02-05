import numpy as np


def make_gate(w, b):
    def gate(x):
        y = np.sum(w * x) + b
        return 0 if y <= 0 else 1
    return gate


AND = make_gate(np.array([1, 1]), -1)
assert AND(np.array([0, 0])) == 0
assert AND(np.array([0, 1])) == 0
assert AND(np.array([1, 0])) == 0
assert AND(np.array([1, 1])) == 1

NAND = make_gate(np.array([-1, -1]), 1.5)
assert NAND(np.array([0, 0])) == 1
assert NAND(np.array([0, 1])) == 1
assert NAND(np.array([1, 0])) == 1
assert NAND(np.array([1, 1])) == 0

OR = make_gate(np.array([1, 1]), 0)
assert OR(np.array([0, 0])) == 0
assert OR(np.array([0, 1])) == 1
assert OR(np.array([1, 0])) == 1
assert OR(np.array([1, 1])) == 1
