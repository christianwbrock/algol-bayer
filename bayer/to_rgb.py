import logging

import numpy as np

logger = logging.getLogger(__name__)


def rawpy_to_rgb(raw):
    """Extract RGB image from a rawpy bayer image."""

    assert all((c in raw.color_desc for c in b'RGB')), 'not a RBG raw image'

    layers = bayer_to_layers(raw.raw_image_visible, raw.raw_pattern)
    return combine_layers_by_color(layers, raw.color_desc, b'RGB')


def combine_layers_by_color(layers, layer_color_desc, target_color_desc=b'RGB', method='mean'):
    """\
    Fold layers by colors.

    Parameters
    ----------
    layers : array_like of shape (N, R, C)
        input image layers

    layer_color_desc : array_like of length N
        A single color for each layer, e.g. 'RGBG'

    target_color_desc: array_like
        A single color for each target layers, e.g. 'RGB'.
        It must only contain elements of layer_color_desc.

    method: str
        'mean', 'median' or any others numpy method of signature method(array, axis=...)
        It is used to combine source layers having the same color, e.g. the two green layers in a RGBG image.

    Return
    ------
    array_like with the same length as target_color_desc
    """
    assert len(layers) == len(layer_color_desc), f'length mismatch between layers and layer_color_desc'

    combiner = getattr(np, method, None)
    assert callable(combiner), f'np.{method} does not exist or is not callable'

    def combine_layers_of_color(color):
        layers_of_correct_color = layers[np.nonzero(np.array(list(layer_color_desc)) == color)]
        return combiner(layers_of_correct_color, axis=0)

    target = [combine_layers_of_color(color) for color in target_color_desc]
    return np.array(target)


def bayer_to_layers(bayer, pattern):
    """\
    Extract color layers from a raw bayer image an a pattern definition.

    If pattern is taken from rawpy.raw_pattern the resulting layers will match rawpy.color_desc.
    E.g. for a Canon D1000 color_desc='RGBG' while raw_pattern=[[0 1],[3 2]]. This results in
    four layers where each layer represents each second pixel, the first red-layer starts at (0,0), the second
    green-layer starts at (0,1) the third blue-layer starts at (1,1) and the forth layer, also green, starts at (1,0).


    Parameters
    ----------

    bayer : array_like of shape (r, c)
        The original raw bayer image

    pattern: array_like
        The smallest possible Bayer pattern of the image.

    """

    number_of_layers = np.max(pattern) + 1
    assert 0 <= np.min(pattern)

    layers = number_of_layers * [None]

    row_step_size, column_step_size = pattern.shape

    indices_y, indices_x = np.indices(pattern.shape)
    for start_row, start_column in zip(indices_y.ravel(), indices_x.ravel()):
        idx = pattern[start_row, start_column]
        layers[idx] = bayer[start_row::row_step_size, start_column::column_step_size]

    return np.asarray(layers)
