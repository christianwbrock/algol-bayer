import logging
import os.path
from argparse import ArgumentParser

import matplotlib.cm
import numpy as np
from matplotlib import pyplot as plt

from bayer.extraction import FastExtraction, find_slit_in_images
from bayer.to_rgb import rawpy_to_rgb
from bayer.utils import multi_glob


def main_raw():
    import rawpy

    logging.basicConfig(level=logging.INFO)

    parser = _create_argument_parser('one or more raw files containing bayer matrices')
    args = parser.parse_args()

    for filename in multi_glob(args.filename):
        if not os.path.exists(filename):
            continue

        with rawpy.imread(filename) as raw:
            layers = rawpy_to_rgb(raw)

        fast = FastExtraction(image_layers=layers, sigma=args.sigma, clipping=args.clipping)
        _plot_file(filename, fast)


def main_fits():
    from astropy.io import fits

    parser = _create_argument_parser('one or more fits files containing images')
    args = parser.parse_args()

    for filename in multi_glob(args.filename):
        with fits.open(filename) as hdu_list:
            images = [hdu.data for hdu in hdu_list if hdu.header.get("NAXIS", 0) == 2]
            if not images:
                logging.error(f"{filename} contains no images")

            for image in images:
                extractor = FastExtraction(image_layers=[image], sigma=args.sigma)
                _plot_file(filename, extractor)


def _plot_file(filename, fast):
    zero_sigma = fast.background_mean
    three_sigma = fast.background_mean + fast.background_stddev * fast.sigma
    ten_sigma = fast.background_mean + fast.background_stddev * fast.clipping

    miny, maxy = find_slit_in_images(fast.layers, zero_sigma, scale=2)

    layers = fast.layers[:, miny:maxy, :]
    num_layers, row_count, column_count = layers.shape

    contour_levels = np.transpose([zero_sigma, three_sigma, ten_sigma])
    contour_colors = 'none green red'.split()  # none sets alpha to zero

    fs = 24
    dpi = 150
    w = column_count / dpi
    h = (fs + num_layers * (fs + row_count)) / dpi

    fig = plt.figure(figsize=(w, h), dpi=dpi)
    fig.canvas.manager.set_window_title(os.path.basename(filename))

    (layer_axes,) = fig.subplots(nrows=num_layers, ncols=1, squeeze=False).T

    for idx_layer in range(num_layers):
        layer_axis = layer_axes[idx_layer]
        layer_axis.get_xaxis().set_visible(False)
        layer_axis.get_yaxis().set_visible(False)

        # title = ((layer_titles[idx_layer] + " - ") if layer_titles else '') \
        #     + f"$\\mu_b={zero_sigma[idx_layer]:.1f}$, $\\sigma_b={fast.background_stddev[idx_layer]:.1f}$"
        # layer_axis.set_title(title)

        layer = layers[idx_layer]

        layer_axis.imshow(layer, cmap=matplotlib.cm.gray)
        layer_axis.contour(np.arange(column_count), np.arange(row_count), layer,
                           levels=contour_levels[idx_layer], colors=contour_colors)

    fig.tight_layout(pad=0.5, h_pad=0.1)
    plt.show()
    plt.close(fig)


def _create_argument_parser(filename_help):
    parser = ArgumentParser(description='Visualize segmentation of an image')
    parser.add_argument('filename', nargs='+', help=filename_help)
    parser.add_argument('--sigma', '-s', default=3.0, type=float, help='sigma used for clipping')
    parser.add_argument('--clipping', '-c', default=10.0, type=float, help='clip background at mean + clipping * stddev')
    return parser
