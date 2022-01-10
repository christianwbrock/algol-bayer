"""\
Use image center-of-mass to extract spectra fast-and-dirty
"""

import logging
import os.path
from argparse import ArgumentParser

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from bayer.to_rgb import rawpy_to_rgb, fits_to_layers
from bayer.utils import multi_glob


def main():
    parser = _create_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    for filename in multi_glob(args.filename):
        if filename.upper().endswith('FIT') or filename.upper().endswith('FITS'):
            import astropy.io.fits as fits

            with fits.open(filename) as hdu_list:
                images = [fits_to_layers(hdu) for hdu in hdu_list if hdu]
                if not images:
                    logging.error(f"{filename} contains no images")

                for image in images:
                    _plot_file(filename, image, args.scale, None)

        else:  # assume raw image
            import rawpy

            for filename in multi_glob(args.filename):
                with rawpy.imread(filename) as raw:
                    _plot_file(filename, rawpy_to_rgb(raw), args.scale, raw.white_level)


def _create_argument_parser():
    parser = ArgumentParser(description='Display fits or raw color or gray level images')
    parser.add_argument('filename', nargs='+', help='one or more files containing images')
    parser.add_argument('--scale', default=False, action='store_true', help='scale image')
    return parser


def _plot_file(filename, image, scale, white_level=None):
    image = np.asarray(image)

    if scale:
        white_level = np.nanmax(image)
    elif white_level is None:
        # let's hope, there is some hot pixel
        maximum = np.nanmax(image)
        white_level = 2 ** np.ceil(np.log2(maximum)) - 1

    image = image / white_level
    num_colors, size_y, size_x = image.shape

    if num_colors == 1:
        image = np.reshape(image, (size_y, size_x))
    else:
        image = np.moveaxis(image, 0, -1)

    fig = plt.figure()
    fig.canvas.manager.set_window_title(os.path.basename(filename))

    ax = fig.add_subplot(1, 1, 1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(image, cmap=matplotlib.cm.gray)

    plt.show()
    plt.close(fig)
