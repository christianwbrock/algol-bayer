"""Display histogram of a raw image."""

import logging
import os.path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import rawpy
from astropy.stats import sigma_clipped_stats

from bayer.to_rgb import rawpy_to_rgb
from bayer.utils import multi_glob


def main():
    parser = ArgumentParser(description='Display histogram of a raw image')
    parser.add_argument('filename', nargs='+', help='one or more raw files containing bayer matrices')
    parser.add_argument('--sigma', '-s', default=3.0, help='sigma used for clipping')
    parser.add_argument('--clipping', '-c', default=10.0, help='clip background at mean + clipping * stddev')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    max_range = 3651

    for filename in multi_glob(args.filename):

        with rawpy.imread(filename) as raw:
            rgb = rawpy_to_rgb(raw)

        fig = plt.figure()
        fig.canvas.set_window_title(os.path.basename(filename))
        ax = fig.add_subplot()

        colors = 'rgb'
        for n, layer in enumerate((rgb[0], rgb[1], rgb[2])):

            (mean, median, stddev) = sigma_clipped_stats(layer, sigma=args.sigma)
            hist = np.histogram(layer[layer >= (mean + args.clipping * stddev)], bins=100)

            ax.plot(hist[1][1:], hist[0], colors[n])

        ax.set_xlabel('intensity')
        ax.set_ylabel('pixel count')
        ax.set_yscale('log')
        ax.axvline(x=max_range, color='k', linestyle='--')
        ax.axvline(x=0.75*max_range, color='k', linestyle='-.')
        fig.tight_layout()
        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    main()

