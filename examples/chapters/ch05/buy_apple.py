from .layers_naive import MultiplyLayer
from .functions import nearly_equal

params = {}
params['price'] = 100
params['count'] = 2
params['tax'] = 1.1

multiply_apple_layer = MultiplyLayer()
apply_tax_layer = MultiplyLayer()

total_price = \
    multiply_apple_layer.forward(params['price'], params['count'])
total_price_including_tax = \
    apply_tax_layer.forward(total_price, params['tax'])
assert nearly_equal(total_price_including_tax, 220)

delta_total_price, delta_tax = \
    apply_tax_layer.backword(1)
assert nearly_equal(delta_total_price, 1.1)
assert nearly_equal(delta_tax, 200)

delta_price, delta_count = \
    multiply_apple_layer.backword(delta_total_price)
assert nearly_equal(delta_price, 2.2)
assert nearly_equal(delta_count, 110)
