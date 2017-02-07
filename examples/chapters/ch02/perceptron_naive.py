def make_gate(w1, w2, b):
    def gate(x1, x2):
        y = w1 * x1 + w2 * x2 + b
        return 0 if y <= 0 else 1
    return gate


AND = make_gate(1, 1, -1)
assert AND(0, 0) == 0
assert AND(0, 1) == 0
assert AND(1, 0) == 0
assert AND(1, 1) == 1

NAND = make_gate(-1, -1, 1.5)
assert NAND(0, 0) == 1
assert NAND(0, 1) == 1
assert NAND(1, 0) == 1
assert NAND(1, 1) == 0

OR = make_gate(1, 1, 0)
assert OR(0, 0) == 0
assert OR(0, 1) == 1
assert OR(1, 0) == 1
assert OR(1, 1) == 1
