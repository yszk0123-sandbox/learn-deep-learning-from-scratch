import numpy as np


def make_gate(w, b):
    def gate(x):
        y = np.sum(w * x) + b
        return 0 if y <= 0 else 1
    return gate


AND = make_gate(np.array([1, 1]), -1)
NAND = make_gate(np.array([-1, -1]), 1.5)
OR = make_gate(np.array([1, 1]), 0)


def XOR(w):
    s1 = NAND(w)
    s2 = OR(w)
    return AND(np.array([s1, s2]))


assert XOR(np.array([0, 0])) == 0
assert XOR(np.array([0, 1])) == 1
assert XOR(np.array([1, 0])) == 1
assert XOR(np.array([1, 1])) == 0
