import numpy as np
from collections import OrderedDict
from .convolution import Convolution
from .pooling import Pooling


class SimpleConvolutionNetwork:
    def __init__(self,
                 input_dimension=(1, 28, 28),
                 convolution_params={
                     'filter_num': 30,
                     'filter_size': 5,
                     'pad': 0,
                     'stride': 1
                 },
                 hidden_size=100,
                 output_size=10,
                 weight_initial_std=0.01):
        filter_num = convolution_params['filter_num']
        filter_size = convolution_params['filter_size']
        filter_pad = convolution_params['pad']
        filter_stride = convolution_params['stride']

        input_size = input_dimension[1]
        convolution_output_size = \
            (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num *
                               (convolution_output_size / 2) *
                               (convolution_output_size / 2))

        self.params = {}
        self.params['W1'] = weight_initial_std * \
            np.random.randn(filter_num, input_dimension[0],
                            filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_initial_std * \
            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_initial_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Convolution1'] = Convolution(self.params['W1'],
                                                  self.params['b1'],
                                                  self.params['stride'],
                                                  self.params['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
