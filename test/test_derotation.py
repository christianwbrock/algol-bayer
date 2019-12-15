import numpy as np
from pytest import approx
from scipy.ndimage.interpolation import rotate

from bayer.fast_extraction import Fast


def test_de_rotation():
    y = 0.2 + 10 * np.exp(-0.5 * (np.arange(0, 30.0 + 1) - 15) ** 2 / 3 ** 2)
    x = 0.2 + 10 * np.exp(-0.5 * (np.arange(0, 100.0 + 1) - 50) ** 2 / 40 ** 2)

    image = np.outer(y, x)

    for expected in np.linspace(-180, 180, 19):
        rotated = rotate(image, angle=-expected, reshape=True, axes=(0, 1))

        fast = Fast(layers=[rotated], sigma=3.0)
        [actual] = fast.de_rotation_angles_deg

        assert (actual == approx(expected, abs=2) or
                actual == approx(expected - 180, abs=2) or
                actual == approx(expected + 180, abs=2))