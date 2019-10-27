import glob

import numpy as np


def bayer_to_rgb(bayer):
    return np.array((bayer[::2, ::2], (bayer[1::2, ::2] + bayer[::2, 1::2]) / 2, bayer[1::2, 1::2]))


def multi_glob(filenames):
    result = []
    for pattern in filenames:
        result.extend(glob.glob(pattern))

    return result