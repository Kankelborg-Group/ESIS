import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from esis.inversion.mart import forward


def test_model():
    n = 64
    azml = 0 * u.deg
    w = [1]
    cube = [np.zeros((1, n, n, n))]
    cube[0][:, n//2, n//2, :] = 1

    d = forward.model(cube, w, azml, 1, input_spatial_offset=(0, 0), output_shape=(1, 2 * n, 2 * n, 1))
    plt.imshow(d.squeeze())
    plt.show()