class MultiplyLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backword(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backword(self, dout):
        return dout, dout


class ReluLayer:
    def __init__(self):
        pass

    def forward(self, x):
        if x > 0:
            return x
        else:
            return 0

    def backword(self, dout):
        if dout > 0:
            return 1
        else:
            return 0
