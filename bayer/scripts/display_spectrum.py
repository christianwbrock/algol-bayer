"""\
Use image center-of-mass to extract spectra fast-and-dirty
"""

import logging
import math
import os.path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from bayer.extraction import FastExtraction
from bayer.to_rgb import rawpy_to_rgb
from bayer.utils import multi_glob


def main_raw():
    import rawpy

    parser = _create_argument_parser('one or more raw file containing bayer matrices')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    for filename in multi_glob(args.filename):
        with rawpy.imread(filename) as raw:
            extractor = FastExtraction(image_layers=rawpy_to_rgb(raw), sigma=args.sigma)
            _plot_file(filename, extractor, raw.white_level, args.cut)


def main_fits():
    from astropy.io import fits

    parser = _create_argument_parser('one or more fits files containing images')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    for filename in multi_glob(args.filename):
        with fits.open(filename) as hdu_list:
            images = [hdu.data for hdu in hdu_list if hdu.header.get("NAXIS", 0) == 2]
            if not images:
                logging.error(f"{filename} contains no images")

            for image in images:
                extractor = FastExtraction(image_layers=[image], sigma=args.sigma)
                _plot_file(filename, extractor, 2**16, args.cut)


def _create_argument_parser(filename_help):
    parser = ArgumentParser(description='Display spectrum from a bayer matrix')
    parser.add_argument('filename', nargs='+', help=filename_help)
    parser.add_argument('--sigma', '-s', default=3.0, help='sigma used for clipping')
    parser.add_argument('--clipping', default=10.0, help='clip background at mean + clipping * stddev')
    cut_group = parser.add_mutually_exclusive_group()
    cut_group.add_argument('--cut', '-c', dest='cut', default=True, action='store_true',
                           help='cut spectrum in dispersion direction (the default)')
    cut_group.add_argument('--dont-cut', '-C', dest='cut', action='store_false',
                           help='do not cut spectrum in dispersion direction')
    return parser


def _plot_file(filename, extractor, white_level, cut_spectra):
    rgb = extractor.de_rotated_layers
    num_colors, size_y, size_x = rgb.shape

    miny, maxy = _find_slit_in_images(rgb, extractor.background_mean)

    if cut_spectra:
        minx, maxx = _find_spectra_in_images(rgb, extractor.background_mean)
    else:
        minx, maxx = 0, size_x - 1

    rgb = rgb[:, miny:maxy, minx:maxx]
    (_, size_y, size_x) = rgb.shape

    xrange = (0, size_x)

    fig = plt.figure()
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

    fig.tight_layout()

    plt.show()
    plt.close(fig)


def _find_slit_in_images(rgb, background_mean):

    __, size_y, __ = rgb.shape
    wo_background = rgb - np.reshape(background_mean, (-1, 1, 1))

    slit_function = np.nanmean(wo_background, axis=(0, 2))
    slit_center, slit_size = _center_of_gravity(slit_function)

    miny = math.floor(slit_center - slit_size)
    maxy = math.ceil(slit_center + slit_size)

    miny = np.clip(miny, 0, size_y - 1)
    maxy = np.clip(maxy, 0, size_y - 1)

    return miny, maxy


def _find_spectra_in_images(rgb, background_mean):

    __, __, size_x = rgb.shape
    wo_background = rgb - np.reshape(background_mean, (-1, 1, 1))

    spectra = np.nanmean(wo_background, axis=1)

    spectrum_locations, spectrum_widths = _center_of_gravity(spectra)

    left_most_spectrum = np.argmin(spectrum_locations)
    right_most_spectrum = np.argmax(spectrum_locations)

    minx = math.floor(spectrum_locations[left_most_spectrum] - spectrum_widths[left_most_spectrum])
    maxx = math.ceil(spectrum_locations[right_most_spectrum] + spectrum_widths[right_most_spectrum])

    minx = np.clip(minx, 0, size_x - 1)
    maxx = np.clip(maxx, 0, size_x - 1)

    return minx, maxx


def _center_of_gravity(data):
    """\
    Return the weighted average and standard deviation.
    """

    if data.ndim > 1:
        result = [_center_of_gravity(data[i]) for i in range(data.shape[0])]
        result = np.transpose(result)
        return result[0], result[1]

    assert data.ndim == 1
    valid_indices = np.isfinite(data)

    indices = np.arange(data.shape[-1])
    indices = indices[valid_indices]
    data = data[valid_indices]

    average = np.average(indices, weights=data)
    # Fast and numerically precise:
    variance = np.average((indices - average) ** 2, weights=data)
    if variance < 0:
        logging.warning('variance < 0?')
        variance = -variance

    return average, math.sqrt(variance)


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
            plt_image[:, :, c] = np.nanmax(plt_image[:, :, c], axis=0)

    return plt_image


if __name__ == '__main__':
    main_raw()
