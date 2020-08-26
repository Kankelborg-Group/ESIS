import numpy as np
import astropy.io.fits

from esis.inversion import mart as mart_


class TestMART:

    def test__call__(self):
        mart = mart_.MART()

        # Need a file
        hdu = astropy.io.fits.open(file)[0]
        data = hdu.data

        r = mart(
            np.zeros((1, 1, 10, 10, 1)),
            cube_shape=(10, 10, 10)
        )
        print(r)
