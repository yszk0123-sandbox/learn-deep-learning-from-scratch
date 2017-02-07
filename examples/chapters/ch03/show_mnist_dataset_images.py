import numpy as np
from PIL import Image
from examples.dataset.mnist import load_mnist


def show_image(image):
    pil_image = Image.fromarray(np.uint8(image))
    pil_image.show()


(x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=False)

image = x_train[0]
label = t_train[0]
reshaped_image = image.reshape(28, 28)

assert image.shape == (784,)
assert reshaped_image.shape == (28, 28)

show_image(reshaped_image)
