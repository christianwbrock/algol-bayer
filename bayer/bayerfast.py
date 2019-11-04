''' Use image center-of-mass to extract spectra fast-and-dirty
'''

import logging
from argparse import ArgumentParser
from os.path import basename

import math
import matplotlib.pyplot as plt
import numpy as np
import rawpy
import warnings

from bayer.utils import multi_glob
from bayer.fast_extraction import Fast

max_range = 3651
border_y = 20


def main():
    warnings.simplefilter("ignore")

    parser = ArgumentParser(description='Display spectrum from a bayer matrix')
    parser.add_argument('filenames', nargs='+', help='one or more raw files containing bayer matrices')
    parser.add_argument('--sigma', '-s', default=10.0, help='sigma used for clipping')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    for filename in multi_glob(args.filenames):

        with rawpy.imread(filename) as raw:

            extractor = Fast(raw.raw_image_visible, sigma=args.sigma)
            # _plot_file(filename, extractor.clipped_de_rotated_rgb)
            _plot_file(filename, extractor.de_rotated_rgb)


def _plot_file(filename, rgb):

        (_, size_y, size_x) = rgb.shape
        center_red, center_green, center_blue = Fast.center_of_gravity(rgb)

        miny = math.floor(np.min((center_red[0], center_green[0], center_blue[0])))
        maxy = math.ceil(np.max((center_red[0], center_green[0], center_blue[0])))

        miny = np.max((0, miny - border_y))
        maxy = np.min((maxy + border_y + 1, size_y - 1))

        minx = math.floor(np.min((center_red[1], center_green[1], center_blue[1])))
        maxx = math.ceil(np.max((center_red[1], center_green[1], center_blue[1])))

        border_x = (maxx - minx) // 2

        minx = 0           # np.max((0, minx - border_x))
        maxx = size_x - 1  # np.min((maxx + border_x + 1, size_x - 1))

        rgb = rgb[:, miny:maxy, minx:maxx]
        (_, size_y, size_x) = rgb.shape

        xrange = (0, size_x)

        plot = plt.subplot2grid((12, 1), (0, 0), rowspan=10)
        plt.title(basename(filename))

        colors = 'rgbk'
        for n, layer in enumerate((rgb[0], rgb[1], rgb[2], rgb[0] + rgb[1] + rgb[2])):
            # spec = np.sum(layer, axis=0)  / np.sum(np.isfinite(layer), axis=0)
            spec = np.nanmax(layer, axis=0)
            plot.plot(range(*spec.shape), spec, colors[n])
            plot.set_xlim(xrange)
            plot.get_xaxis().set_visible(False)

        plt.axhline(y=max_range, color='k', linestyle='--')
        plt.axhline(y=0.75*max_range, color='k', linestyle='-.')

        plot = plt.subplot2grid((12, 1), (10, 0))
        plot.imshow(_reshape(rgb, scale=False), aspect='auto')
        plot.set_xlim(xrange)
        plot.get_xaxis().set_visible(False)
        plot.get_yaxis().set_visible(False)

        plot = plt.subplot2grid((12, 1), (11, 0))
        plot.imshow(_reshape(rgb, scale=True), aspect='auto')
        plot.set_xlim(xrange)
        plot.get_xaxis().set_visible(False)
        plot.get_yaxis().set_visible(False)

        plt.tight_layout(h_pad=0)

        plt.show()


def _reshape(data, scale=False):

    assert data.shape[0] == 3
    img = np.moveaxis(data, 0, 2)
    assert img.shape[2] == 3

    if scale:
        img = img / np.nanmax(data)

        img[:,:,0] = np.nanmax(img[:,:,0], axis=(0))
        img[:,:,1] = np.nanmax(img[:,:,1], axis=(0))
        img[:,:,2] = np.nanmax(img[:,:,2], axis=(0))
    else:
        img = img / max_range

    return img


if __name__ == '__main__':
    main()

