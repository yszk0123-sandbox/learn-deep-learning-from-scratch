import numpy as np

A = np.array([10, 20, 30, 40])
assert np.ndim(A) == 1
assert A.shape == (4,)

B = np.array([[10, 20], [30, 40], [50, 60]])
assert np.ndim(B) == 2
assert B.shape == (3, 2)
