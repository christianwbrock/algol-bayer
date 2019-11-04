import math
import numpy as np
import rawpy
from pytest import approx
from scipy.ndimage.interpolation import rotate

from fast_extraction import Fast


def test_covariance():

    filenames = [
        '/home/guest/Documents/Astro/data/2015-02-20-betori/orig/IMG_2375.CR2',
        '/home/guest/Documents/Astro/data/2015-02-19-alpori/orig/IMG_2236.CR2'
    ]

    for filename in filenames:
        with rawpy.imread(filename) as raw:
            fast = Fast(bayer=raw.raw_image_visible)

            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(211)
            ax.imshow(np.moveaxis(fast.rgb, 0, 2) / 4000)
            ax = fig.add_subplot(212)
            ax.imshow(np.moveaxis(fast.de_rotated_rgb, 0, 2) / 4000)
            plt.show()


def test_de_rotation():

    y = _normal(max=1, loc=15, scale=3, size=30)
    x = _normal(max=1, loc=50, scale=40, size=100)

    image = np.outer(y, x)

    for angle in np.linspace(-180, 180, 19):
        rotated = rotate(image, angle=-angle, reshape=True, axes=(0, 1))

        fast = Fast(layers=[rotated], sigma=None)

        assert fast.de_rotation_angles_deg == approx(angle, abs=2) or \
               fast.de_rotation_angles_deg == approx(angle - 180, abs=2) or \
               fast.de_rotation_angles_deg == approx(angle + 180, abs=2)


def _normal(max, loc, scale, size):
    x = np.arange(0, size + 1, 1)
    return max * np.exp(-(x - loc) ** 2 / 2 / scale / scale)
