import glob

import numpy as np


def bayer_to_rgb(bayer, color_desc=b'RGBG', raw_color_pattern=[0, 1, 2, 3]):

    layers = bayer_to_rgbg(bayer)
    assert 4 == len(color_desc) == len(raw_color_pattern) == layers.shape[0]

    # reorder layers according to the color pattern
    # TODO does no work for Nikon NEF files
    layers = layers[raw_color_pattern]

    # for layer for each color
    def mean(c):
        layers_of_color = layers[np.nonzero(np.array(list(color_desc)) == list(c)[0])]
        return np.mean(layers_of_color, axis=0)

    r = mean(b'R')
    g = mean(b'G')
    b = mean(b'B')

    return np.array([r, g, b])


def bayer_to_rgbg(bayer):
    return np.array((bayer[::2, ::2], bayer[::2, 1::2], bayer[1::2, ::2], bayer[1::2, 1::2]))


def multi_glob(filenames):
    result = []
    for pattern in filenames:
        result.extend(glob.glob(pattern))

    return result
