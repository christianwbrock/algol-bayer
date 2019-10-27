''' Try how rawpy is doing for raw image calculation stuff

'''
import glob
import logging
from argparse import ArgumentParser
from os.path import basename

import matplotlib.pyplot as plt
import numpy as np
import rawpy
from astropy.stats import sigma_clipped_stats

from bayer.utils import bayer_to_rgb, multi_glob


# import rawpy.enhance


def main():
    parser = ArgumentParser(description='Display histogram of a raw image')
    parser.add_argument('filenames', nargs='+', help='one or more raw files containing bayer matrices')
    parser.add_argument('--sigma', '-s', default=3.0, help='sigma used for clipping')
    parser.add_argument('--clipping', '-c', default=10.0, help='sigma used for clipping')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    max_range = 3651

    for filename in multi_glob(args.filenames):

        with rawpy.imread(filename) as raw:
            rgb = bayer_to_rgb(raw.raw_image_visible)

            colors = 'rgb'
            for n, layer in enumerate((rgb[0], rgb[1], rgb[2])):

                (mean, median, stddev) = sigma_clipped_stats(layer, sigma=args.sigma)
                hist = np.histogram(layer[layer >= (mean + args.clipping * stddev)], bins=100)

                plt.plot(hist[1][1:], hist[0], colors[n])

            plt.title("%s" % filename)
            plt.xlabel('intensity')
            plt.ylabel('log')
            plt.yscale('log')
            plt.axvline(x=max_range, color='k', linestyle='--')
            plt.axvline(x=0.75*max_range, color='k', linestyle='-.')
            plt.show()


if __name__ == '__main__':
    main()

