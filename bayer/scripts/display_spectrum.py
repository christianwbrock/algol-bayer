"""\
Use image center-of-mass to extract spectra fast-and-dirty
"""

import logging
import os.path
import warnings
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from bayer.extraction import FastExtraction
from bayer.extraction import find_slit_in_images
from bayer.to_rgb import rawpy_to_rgb, fits_to_layers
from bayer.utils import multi_glob


def main_raw():
    import rawpy

    parser = _create_argument_parser('one or more raw file containing bayer matrices')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    for filename in multi_glob(args.filename):
        with rawpy.imread(filename) as raw:
            extractor = FastExtraction(image_layers=rawpy_to_rgb(raw), sigma=args.sigma)
            _plot_file(filename, extractor, raw.white_level, args.store)


def main_fits():
    from astropy.io import fits

    parser = _create_argument_parser('one or more fits files containing images')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    for filename in multi_glob(args.filename):
        with fits.open(filename) as hdu_list:
            images = [fits_to_layers(hdu) for hdu in hdu_list if hdu]
            if not images:
                logging.error(f"{filename} contains no images")

            for image in images:
                extractor = FastExtraction(image_layers=image, sigma=args.sigma)
                # TODO 2**BITPIX
                _plot_file(filename, extractor, 2 ** 16, args.store)


def _create_argument_parser(filename_help):
    parser = ArgumentParser(description='Display spectrum from a bayer matrix')
    parser.add_argument('filename', nargs='+', help=filename_help)
    parser.add_argument('--sigma', '-s', default=3.0, type=float, help='sigma used for clipping')
    parser.add_argument('--clipping', default=10.0, type=float, help='clip background at mean + clipping * stddev')
    parser.add_argument('--store', metavar='output.png', help='Store plot as file.')
    return parser


def _plot_file(filename, extractor, white_level, store):
    rgb = extractor.de_rotated_layers

    miny, maxy = find_slit_in_images(rgb, extractor.background_mean)

    rgb = rgb[:, miny:maxy, :]
    num_colors, size_y, size_x = rgb.shape

    xrange = (0, size_x)

    dpi = 150

    figsize_x = (size_x * 1.5) / dpi + 1

    fig = plt.figure(figsize=(figsize_x, figsize_x / 2), dpi=dpi)
    fig.canvas.manager.set_window_title(os.path.basename(filename))

    ax = plt.subplot2grid((12, 8), (0, 0), rowspan=10, colspan=7)
    ax.set_xlim(xrange)
    ax.get_xaxis().set_visible(False)

    if num_colors == 3:
        layers = rgb[0], rgb[1], rgb[2], rgb[0] + rgb[1] + rgb[2]
        colors = 'rgbk'
    else:
        layers = rgb
        colors = 'k' * num_colors

    for color, layer in zip(colors, layers):
        # In some columns all values maybe nan, this can be ignored
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            spec = np.nanmax(layer, axis=0)
        ax.plot(range(*spec.shape), spec, color)

    ax.axhline(y=white_level, color='k', linestyle='--')
    ax.axhline(y=0.75 * white_level, color='k', linestyle='-.')

    ax = plt.subplot2grid((12, 8), (10, 7))
    slit = np.nanmax(rgb, axis=(0, 2))
    ax.plot(slit, range(*slit.shape), 'k')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot2grid((12, 8), (10, 0), colspan=7)
    ax.imshow(_reshape_and_scale_image(rgb, white_level, scale=False), aspect='auto')
    ax.set_xlim(xrange)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot2grid((12, 8), (11, 0), colspan=7)
    ax.imshow(_reshape_and_scale_image(rgb, white_level, scale=True), aspect='auto')
    ax.set_xlim(xrange)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    fig.tight_layout(pad=0.5, h_pad=0.2, w_pad=0.2)

    if store:
        plt.savefig(store)
    else:
        plt.show()
    plt.close(fig)


def _reshape_and_scale_image(data, max_camera_white_level, scale=False):
    """\
    plt.imshow() expects the image to have shape (size_y, size_x, num_colors) and values in [0.0 .. 1.0].
    """
    num_colors, size_y, size_x = data.shape
    plt_image = np.moveaxis(data, 0, 2)
    assert plt_image.shape == (size_y, size_x, num_colors)

    plt_image /= max_camera_white_level

    if scale:
        # brighten image so the maximum becomes 1
        plt_image /= np.nanmax(data)

        # and replace each y-column with it's maximum
        for c in range(num_colors):
            # In some columns all values maybe nan, this can be ignored
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                plt_image[:, :, c] = np.nanmax(plt_image[:, :, c], axis=0)

    return plt_image
