import numpy as np
from examples.common.util import im2col


class PoolingLayer:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col_x = im2col(x,
                       self.pool_h,
                       self.pool_w,
                       stride=self.stride,
                       pad=self.pad)
        col_x = col_x.reshape(-1, self.pool_h * self.pool_w)

        out = np.max(col_x, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out