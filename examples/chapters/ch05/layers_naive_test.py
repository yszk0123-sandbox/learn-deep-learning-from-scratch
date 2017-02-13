from .layers_naive import MultiplyLayer, AddLayer, ReluLayer
from .functions import nearly_equal


def test_multiply_layer():
    multiply_layer = MultiplyLayer()
    z = multiply_layer.forward(2, 3)
    dx, dy = multiply_layer.backword(1)
    assert nearly_equal(z, 6)
    assert nearly_equal(dx, 3)
    assert nearly_equal(dy, 2)


def test_add_layer():
    add_layer = AddLayer()
    z = add_layer.forward(2, 3)
    dx, dy = add_layer.backword(1)
    assert nearly_equal(z, 5)
    assert nearly_equal(dx, 1)
    assert nearly_equal(dy, 1)


def test_relu_layer():
    relu_layer = ReluLayer()
    z = relu_layer.forward(0)
    assert nearly_equal(z, 0)
    assert nearly_equal(relu_layer.backword(-1), 0)
    assert nearly_equal(relu_layer.backword(0), 0)
    assert nearly_equal(relu_layer.backword(3), 1)

    relu_layer = ReluLayer()
    z = relu_layer.forward(-1)
    assert nearly_equal(z, 0)

    relu_layer = ReluLayer()
    z = relu_layer.forward(3)
    assert nearly_equal(z, 3)


test_multiply_layer()
test_add_layer()
test_relu_layer()
