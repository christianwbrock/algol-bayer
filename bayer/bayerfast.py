''' Use image center-of-mass to extract spectra fast-and-dirty
'''

import logging
from argparse import ArgumentParser
from os.path import basename

import math
import matplotlib.pyplot as plt
import numpy as np
import rawpy
from scipy.ndimage.interpolation import rotate
from astropy.stats import sigma_clipped_stats

from bayer.utils import bayer_to_rgb, multi_glob

max_range = 3651
border_y = 20


def main():
    parser = ArgumentParser(description='Display spectrum from a bayer matrix')
    parser.add_argument('filenames', nargs='+', help='one or more raw files containing bayer matrices')
    parser.add_argument('--sigma', '-s', default=3.0, help='sigma used for clipping')
    parser.add_argument('--clipping', '-c', default=100.0, help='sigma used for clipping')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    for filename in multi_glob(args.filenames):

        with rawpy.imread(filename) as raw:
            rgb = bayer_to_rgb(raw.raw_image_visible)

            try_again = True
            clipping = args.clipping

            while try_again and clipping > args.sigma:
                try:
                    _plot_file(filename, rgb, args.sigma, clipping)
                    try_again = False
                except Exception as e:
                    logging.error("Failed to extract spectrum from %s: %s", basename(filename), e)
                    clipping /= 2


def _plot_file(filename, rgb, sigma, clipping):

        rgb, center_red, center_green, center_blue = _rotate_horizontally(rgb, sigma, clipping)
        (_, size_y, size_x) = rgb.shape

        miny = math.floor(np.min((center_red[0], center_green[0], center_blue[0])))
        maxy = math.ceil(np.max((center_red[0], center_green[0], center_blue[0])))

        miny = np.max((0, miny - border_y))
        maxy = np.min((maxy + border_y + 1, size_y - 1))

        miny = np.max((0, miny - border_y))
        maxy = np.min((maxy + border_y + 1, size_y - 1))

        minx = math.floor(np.min((center_red[1], center_green[1], center_blue[1])))
        maxx = math.ceil(np.max((center_red[1], center_green[1], center_blue[1])))

        border_x = (maxx - minx) // 2

        minx = np.max((0, minx - border_x))
        maxx = np.min((maxx + border_x + 1, size_x - 1))

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


def _rotate_horizontally(rgb, sigma, clipping):

    while True:

        center_red = _compute_center(rgb[0], sigma, clipping)
        center_green = _compute_center(rgb[1], sigma, clipping)
        center_blue = _compute_center(rgb[2], sigma, clipping)

        dy = center_red[0] - center_blue[0]
        dx = center_red[1] - center_blue[1]

        angle_deg = math.degrees(math.atan2(dy, dx))

        if math.fabs(angle_deg) < 1:
            return rgb, center_red, center_green, center_blue

        rgb = rotate(rgb, angle_deg, axes=(1, 2), mode='constant', cval=np.nan)


def _reshape(data, scale=False):

    assert data.shape[0] == 3
    img = np.moveaxis(data, 0, 2)
    assert img.shape[2] == 3

    if scale:
        img /= np.nanmax(data)
        img[:,:,0] = np.nanmax(img[:,:,0], axis=(0))
        img[:,:,1] = np.nanmax(img[:,:,1], axis=(0))
        img[:,:,2] = np.nanmax(img[:,:,2], axis=(0))
    else:
        img /= max_range

    return img


def _compute_center(data, sigma, clipping):

    assert np.ndim(data) == 2

    data = np.ma.array(data, fill_value=np.nan)
    data[np.isnan(data)] = np.ma.masked
    data[np.isinf(data)] = np.ma.masked

    (mean, median, stddev) = sigma_clipped_stats(data, sigma=sigma, cenfunc='mean')

    with np.errstate(invalid='ignore'):
        data[data < (mean + clipping)] = np.ma.masked

    indices = np.indices(data.shape)

    center = np.sum(data*indices, axis=(1,2)) / np.sum(data)
    return center


if __name__ == '__main__':
    main()

