import glob

import numpy as np


def multi_glob(filenames):
    result = []
    for pattern in filenames:
        result.extend(glob.glob(pattern))

    return result


def fits_white_level(header, data=None):
    """\
    Return the maximum digital level for a FITS image.

    Uses BITPIX when a header is given and BITPIX is a positive integer
    (2**BITPIX - 1). If header is None, BITPIX is missing, or BITPIX is
    floating-point, estimates from data when provided, otherwise falls back
    to 16-bit full scale.
    """
    if header is not None:
        bitpix = header.get('BITPIX')
        if bitpix is not None:
            bitpix = int(bitpix)
            if bitpix > 0:
                return 2 ** bitpix - 1

    if data is not None:
        maximum = np.nanmax(data)
        if np.isfinite(maximum) and maximum > 0:
            return 2 ** int(np.ceil(np.log2(maximum + 1))) - 1

    return 2 ** 16 - 1
