import glob

import numpy as np

import logging
logger = logging.getLogger(__name__)


def rawpy_to_rgb(raw):
    """Extract RGB image from a rawpy bayer image."""

    assert all((c in raw.color_desc for c in b'RGB')), 'not an RBG raw image'

    layers = bayer_to_layers(raw.raw_image_visible, raw.raw_pattern)
    return combine_layers_by_color(layers, raw.color_desc, b'RGB')


def combine_layers_by_color(layers, layer_color_desc, target_color_desc=b'RGB', method='mean'):
    """Fold layers by colors.

    Parameters
    ----------
    layers : array_like of shape (N, R, C)
        input image layers

    layer_color_desc : array_like of length N
        A single color for each layer

    target_color_desc: array_like
        A single color for each target layers. It must only contain elements of layer_color_desc.

    method: str
        'mean', 'median' or any others numpy method of signature method(array, axis=...)

    Return
    ------
    array_like with the same length as target_color_desc
    """
    assert len(layers) == len(layer_color_desc), f'length mismatch between layers and layer_color_desc'

    combiner = getattr(np, method, None)
    assert callable(combiner), f'np.{method} does not exist or is not callable'

    def combine(color):
        layers_of_correct_color = layers[np.nonzero(np.array(list(layer_color_desc)) == color)]
        return combiner(layers_of_correct_color, axis=0)

    target = []
    for c in target_color_desc:
        target.append(combine(c))

    return np.array(target)


def bayer_to_layers(bayer, pattern):
    """Extract """

    number_of_layers = np.max(pattern) + 1
    assert 0 <= np.min(pattern)

    layers = number_of_layers * [None]

    row_count, column_count = pattern.shape
    indices = np.indices(pattern.shape)
    indices = np.moveaxis(indices, 0, -1)
    indices = np.reshape(indices, (-1, 2))
    for r, c in indices:
        idx = pattern[r, c]
        layers[idx] = bayer[r::row_count, c::column_count]

    return np.asarray(layers)


def multi_glob(filenames):
    result = []
    for pattern in filenames:
        result.extend(glob.glob(pattern))

    return result
