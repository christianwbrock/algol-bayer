import numpy as np
import pytest

from bayer.to_rgb import bayer_to_layers

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
