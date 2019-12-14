''' Use image center-of-mass to extract spectra fast-and-dirty
'''

import logging
import warnings
from argparse import ArgumentParser
from os.path import basename

import math
import matplotlib.pyplot as plt
import numpy as np
import rawpy
from astropy.stats import sigma_clipped_stats

from bayer.fast_extraction import Fast
from bayer.utils import multi_glob

max_range = 3651
border_y = 20


def main():
    warnings.simplefilter("ignore")

    parser = ArgumentParser(description='Display spectrum from a bayer matrix')
    parser.add_argument('filenames', nargs='+', help='one or more raw files containing bayer matrices')
    parser.add_argument('--sigma', '-s', default=10.0, type=float, help='sigma used for clipping')
    cut_group = parser.add_mutually_exclusive_group()
    cut_group.add_argument('--cut', '-c', dest='cut', default=True, action='store_true', help='cut spectrum in dispersion direction (the default)')
    cut_group.add_argument('--dont-cut', '-C', dest='cut', action='store_false', help='do not cut spectrum in dispersion direction')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    for filename in multi_glob(args.filenames):

        with rawpy.imread(filename) as raw:

            extractor = Fast(raw, sigma=args.sigma)
            # _plot_file(filename, extractor.clipped_de_rotated_rgb)
            _plot_file(filename, extractor, args.cut)


def _plot_file(filename, extractor, cut):

    rgb = extractor.de_rotated_rgb
    num_colors, size_y, size_x = rgb.shape

    y_values = np.nanmean(rgb - np.reshape(extractor.background_median, (num_colors, 1, 1)), axis=(0, 2))
    y_indices = np.arange(0, y_values.shape[0])

    def weighted_avg_and_std(values, weights):
        """
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        """

        filter = np.isfinite(values) & np.isfinite(weights)

        values = values[filter]
        weights = weights[filter]

        average = np.average(values, weights=weights)
        # Fast and numerically precise:
        variance = np.average((values - average) ** 2, weights=weights)
        if variance < 0:
            logging.warning('variance < 0?')
            variance = -variance

        return (average, math.sqrt(variance))

    moments_y_avg = weighted_avg_and_std(y_indices, y_values)

    miny = math.floor(moments_y_avg[0] - moments_y_avg[1])
    maxy = math.ceil(moments_y_avg[0] + moments_y_avg[1])

    miny = np.max((0, miny))
    maxy = np.min((maxy, size_y - 1))

    if cut:
        x_values = np.nanmean(rgb - np.reshape(extractor.background_median, (num_colors, 1, 1)), axis=(1))
        x_indices = np.arange(0, x_values.shape[1])

        moments_x_rgb = np.asarray([weighted_avg_and_std(x_indices, x_values[n]) for n in range(num_colors)])

        index_min = np.argmin(moments_x_rgb[:,0])
        index_max = np.argmax(moments_x_rgb[:,0])

        minx = math.floor(moments_x_rgb[index_min][0] - moments_x_rgb[index_min][1])
        maxx = math.ceil(moments_x_rgb[index_max][0] + moments_x_rgb[index_max][1])

        minx = np.max((0, minx))
        maxx = np.min((maxx, size_x - 1))
    else:
        minx = 0
        maxx = size_x - 1

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

