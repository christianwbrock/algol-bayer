import os

import numpy as np
import rawpy
from pytest import approx
from scipy.ndimage.interpolation import rotate

from bayer.fast_extraction import Fast


def test_covariance():

    filenames = [
        '/home/guest/Documents/Astro/data/2015-02-20-betori/orig/IMG_2375.CR2',
        '/home/guest/Documents/Astro/data/2015-02-19-alpori/orig/IMG_2236.CR2'
    ]

    for filename in filenames:
        if not os.path.exists(filename):
            continue
        with rawpy.imread(filename) as raw:
            fast = Fast(bayer=raw.raw_image_visible)

            # use the functional interface of pyplot ;-)
            import matplotlib.pyplot as plt
            plt.subplot(211)
            plt.imshow(np.moveaxis(fast.rgb, 0, 2) / 4000)
            plt.subplot(212)
            plt.imshow(np.moveaxis(fast.de_rotated_rgb, 0, 2) / 4000)
            plt.show()
            plt.clf()


def test_de_rotation():
    y = _normal(max_value=1, loc=15, scale=3, size=30, background=0.2)
    x = _normal(max_value=1, loc=50, scale=40, size=100, background=0.2)

    image = np.outer(y, x)

    for expected in np.linspace(-180, 180, 19):
        rotated = rotate(image, angle=-expected, reshape=True, axes=(0, 1))

        fast = Fast(layers=[rotated], sigma=3.0)
        actual = fast.de_rotation_angles_deg

        assert (actual == approx(expected, abs=2) or
                actual == approx(expected - 180, abs=2) or
                actual == approx(expected + 180, abs=2))


def _normal(max_value, loc, scale, size, background):
    x = np.arange(0, size + 1, 1)
    return background + max_value * np.exp(-(x - loc) ** 2 / 2 / scale / scale)
