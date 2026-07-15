import numpy as np
from pytest import approx

from bayer.extraction import FastExtraction, find_slit_in_images


def _synthetic_spectrum_image(spectrum_center=100, slit_center=15):
    spectrum = 0.2 + 10 * np.exp(-0.5 * (np.arange(201) - spectrum_center) ** 2 / 40 ** 2)
    slit = 0.2 + 10 * np.exp(-0.5 * (np.arange(31) - slit_center) ** 2 / 3 ** 2)
    image = np.outer(slit, spectrum)
    return np.stack([image, image, image])


def test_find_slit_on_synthetic_image():
    layers = _synthetic_spectrum_image(slit_center=15)
    miny, maxy = find_slit_in_images(layers, np.zeros(3))
    assert (miny + maxy) / 2 == approx(15, abs=2)


def test_spectrum_peak_on_synthetic_image():
    layers = _synthetic_spectrum_image(spectrum_center=100, slit_center=15)
    fast = FastExtraction(image_layers=layers, sigma=3.0)

    rgb = fast.de_rotated_layers
    miny, maxy = find_slit_in_images(rgb, fast.background_mean)
    cropped = rgb[:, miny:maxy, :]

    spec = np.nanmax(cropped[0], axis=0)
    assert np.nanargmax(spec) == 100
