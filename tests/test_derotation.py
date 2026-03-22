import numpy as np
from pytest import approx
from scipy.ndimage import rotate

from bayer.extraction import FastExtraction


def test_de_rotation():
    slit = 0.2 + 10 * np.exp(-0.5 * (np.arange(0, 30.0 + 1) - 15) ** 2 / 3 ** 2)
    spectrum = 0.2 + 10 * np.exp(-0.5 * (np.arange(0, 200.0 + 1) - 100) ** 2 / 40 ** 2)

    image = np.outer(slit, spectrum)

    for expected in np.arange(-180, 181, 20):
        rotated = rotate(image, angle=-expected, reshape=True, axes=(0, 1))

        fast = FastExtraction(image_layers=[rotated], sigma=3.0)
        [actual] = fast.de_rotation_angles_deg

        assert (actual == approx(expected, abs=2) or
                actual == approx(expected - 180, abs=2) or
                actual == approx(expected + 180, abs=2))
