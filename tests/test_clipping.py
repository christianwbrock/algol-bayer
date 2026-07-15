import numpy as np

from bayer.extraction import FastExtraction


def _synthetic_layers():
    spectrum = 0.2 + 10 * np.exp(-0.5 * (np.arange(201) - 100) ** 2 / 40 ** 2)
    slit = 0.2 + 10 * np.exp(-0.5 * (np.arange(31) - 15) ** 2 / 3 ** 2)
    image = np.outer(slit, spectrum)
    return np.stack([image, image, image])


def test_higher_clipping_removes_more_pixels():
    layers = _synthetic_layers()
    low_threshold = FastExtraction(layers, sigma=3.0, clipping=3.0)
    high_threshold = FastExtraction(layers, sigma=3.0, clipping=20.0)
    assert np.sum(np.isfinite(high_threshold.clipped_layers)) < np.sum(
        np.isfinite(low_threshold.clipped_layers))
