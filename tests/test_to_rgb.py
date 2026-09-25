import numpy as np
import pytest
from astropy.io import fits

from bayer.to_rgb import bayer_to_layers, fits_to_layers

PATTERN = [[0, 1], [2, 3]]


@pytest.mark.parametrize('rows, columns', [(4, 6), (5, 6), (4, 7), (5, 7)])
def test_bayer_to_layers_crops_partial_patterns(rows, columns):
    bayer = np.arange(rows * columns).reshape(rows, columns)
    cropped = bayer[:rows - rows % 2, :columns - columns % 2]

    layers = bayer_to_layers(bayer, PATTERN)

    assert layers.shape == (4, cropped.shape[0] // 2, cropped.shape[1] // 2)
    np.testing.assert_array_equal(layers[0], cropped[0::2, 0::2])
    np.testing.assert_array_equal(layers[1], cropped[0::2, 1::2])
    np.testing.assert_array_equal(layers[2], cropped[1::2, 0::2])
    np.testing.assert_array_equal(layers[3], cropped[1::2, 1::2])


def test_bayer_to_layers_keeps_colors_of_odd_sized_mosaic():
    bayer = np.tile([[10, 20], [30, 40]], (3, 4))[:-1, :-1]
    assert bayer.shape == (5, 7)

    layers = bayer_to_layers(bayer, PATTERN)

    assert layers.shape == (4, 2, 3)
    for layer, color in zip(layers, [10, 20, 30, 40]):
        assert np.all(layer == color)


def _rggb_hdu(x_off=0, y_off=0):
    mosaic = np.tile([[10.0, 20.0], [30.0, 40.0]], (2, 2))
    header = fits.Header({
        'BAYERPAT': 'RGGB',
        'XBAYROFF': x_off,
        'YBAYROFF': y_off,
    })
    return fits.PrimaryHDU(mosaic, header=header)


def test_fits_to_layers_zero_offset_matches_missing_offset_keywords():
    mosaic = np.tile([[10.0, 20.0], [30.0, 40.0]], (2, 2))
    with_zero = _rggb_hdu(0, 0)
    without = fits.PrimaryHDU(mosaic, header=fits.Header({'BAYERPAT': 'RGGB'}))

    np.testing.assert_array_equal(fits_to_layers(with_zero), fits_to_layers(without))
    np.testing.assert_array_equal(fits_to_layers(with_zero)[0], 10)
    np.testing.assert_array_equal(fits_to_layers(with_zero)[2], 40)


@pytest.mark.parametrize('x_off, y_off, red, blue', [
    (1, 0, 20, 30),
    (0, 1, 30, 20),
    (1, 1, 40, 10),
])
def test_fits_to_layers_origin_offsets_shift_rgb_quadrants(x_off, y_off, red, blue, caplog):
    with caplog.at_level('INFO', logger='bayer.to_rgb'):
        layers = fits_to_layers(_rggb_hdu(x_off, y_off))

    np.testing.assert_array_equal(layers[0], red)
    np.testing.assert_array_equal(layers[2], blue)
    assert f'XBAYROFF={x_off}' in caplog.text
    assert f'YBAYROFF={y_off}' in caplog.text
