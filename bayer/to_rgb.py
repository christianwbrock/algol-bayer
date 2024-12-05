import logging

import numpy as np

logger = logging.getLogger(__name__)


def fits_to_layers(fits):
    """\
    Having a fits image, convert it into a RGB three layer image or  a single layer gray-scale-image.

    :returns None if something fails
    """
    hdr = fits.header

    naxis = hdr.get("NAXIS", 0)
    data = np.asarray(fits.data, dtype=np.float32)
    if naxis == 3:
        return data

    if naxis != 2:
        return None

    # only RGB images have a bayer pattern (?)
    bayer_pattern = hdr.get('BAYERPAT')
    if not bayer_pattern:
        return [data]

    # 0123 has been validated w/ a Meade DSI IV on KStars
    layers = bayer_to_layers(data, [[0, 1], [2, 3]])
    return combine_layers_by_color(layers, bayer_pattern, b'RGB')


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
    assert len(layers) == len(layer_color_desc), 'length mismatch between layers and layer_color_desc'

    if isinstance(layer_color_desc, str):
        layer_color_desc = layer_color_desc.encode("UTF-8")

    combiner = getattr(np, method, None)
    assert callable(combiner), f'np.{method} does not exist or is not callable'

    def _combine_layers_of_color(color):
        layers_of_correct_color = layers[np.nonzero(np.array(list(layer_color_desc)) == color)]
        return combiner(layers_of_correct_color, axis=0)

    target = [_combine_layers_of_color(color) for color in target_color_desc]
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

    pattern = np.asarray(pattern)
    number_of_layers = np.max(pattern) + 1
    assert 0 <= np.min(pattern)

    layers = number_of_layers * [None]

    row_step_size, column_step_size = pattern.shape
    rows, columns = np.shape(bayer)
    if rows % row_step_size != 0 or columns % column_step_size != 0:
        bayer = np.resize(bayer,
                          (rows // row_step_size * row_step_size, columns // column_step_size * column_step_size))

    indices_y, indices_x = np.indices(pattern.shape)
    for start_row, start_column in zip(indices_y.ravel(), indices_x.ravel()):
        idx = pattern[start_row, start_column]
        layers[idx] = bayer[start_row::row_step_size, start_column::column_step_size]

    return np.asarray(layers)
