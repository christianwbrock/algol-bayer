"""Display histogram of a raw image."""

import logging
import os.path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clipped_stats

from bayer.to_rgb import fits_to_layers, rawpy_to_rgb
from bayer.utils import fits_white_level, multi_glob


def main_raw():
    import rawpy

    parser = _create_argument_parser('one or more raw files containing bayer matrices')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    for filename in multi_glob(args.filename):
        with rawpy.imread(filename) as raw:
            rgb = rawpy_to_rgb(raw)
            max_range = raw.white_level

        _plot_histogram(os.path.basename(filename), rgb, max_range, args.sigma, args.clipping)


def main_fits():
    from astropy.io import fits

    parser = _create_argument_parser('one or more fits files containing images')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    for filename in multi_glob(args.filename):
        with fits.open(filename) as hdu_list:
            images = [
                (fits_to_layers(hdu), hdu.header)
                for hdu in hdu_list if hdu]
            images = [
                (image, header)
                for image, header in images if image is not None]
            if not images:
                logging.error(f"{filename} contains no images")
                continue

            for image, header in images:
                image = np.asarray(image)
                max_range = fits_white_level(header, image)
                _plot_histogram(
                    os.path.basename(filename), image, max_range,
                    args.sigma, args.clipping)


def _create_argument_parser(filename_help=None):
    parser = ArgumentParser(description='Display histogram of a raw image')
    parser.add_argument('filename', nargs='+', help=filename_help)
    parser.add_argument('--sigma', '-s', default=3.0, type=float, help='sigma used for clipping')
    parser.add_argument('--clipping', '-c', default=10.0, type=float,
                        help='clip background at mean + clipping * stddev')
    return parser


def _plot_histogram(title, layers, max_range, sigma, clipping):
    fig = plt.figure()
    fig.canvas.manager.set_window_title(title)

    num_layers, __, __ = layers.shape

    ax = fig.add_subplot()
    if num_layers == 3:
        colors = 'rgb'
    else:
        colors = 'k' * num_layers

    for color, layer in zip(colors, [layers[i] for i in range(num_layers)]):
        (mean, median, stddev) = sigma_clipped_stats(layer, sigma=sigma)
        hist = np.histogram(layer[layer >= (mean + clipping * stddev)], bins=100)

        ax.plot(hist[1][1:], hist[0], color)

    ax.set_xlabel('intensity')
    ax.set_ylabel('pixel count')
    ax.set_yscale('log')
    ax.axvline(x=max_range, color='k', linestyle='--')
    ax.axvline(x=0.75 * max_range, color='k', linestyle='-.')
    fig.tight_layout(pad=0.5, h_pad=0.2, w_pad=0.2)
    plt.show()
    plt.close(fig)
