''' Try how rawpy is doing for raw image calculation stuff

'''
import rawpy
# import rawpy.enhance

import numpy as np
from os.path import basename
from argparse import ArgumentParser
import logging
import matplotlib.pyplot as plt


def main():
    parser = ArgumentParser(description='Display histogram of a raw image')
    parser.add_argument('filename', help='raw file name containing bayer matrix')
    parser.add_argument('--percentile', '-p', default=99.99, help='percentile displayed in the title')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    max_range = 2**12

    with rawpy.imread(args.filename) as raw:
        data = raw.raw_image_visible

        info = (basename(args.filename), np.min(data), np.mean(data), np.percentile(data, args.percentile), np.max(data))

        red = np.histogram(data[::2][::2], bins=256, range=[0, max_range])
        gr1 = np.histogram(data[1::2][::2], bins=256, range=[0, max_range])
        gr2 = np.histogram(data[::2][1::2], bins=256, range=[0, max_range])
        assert len(gr1[0]) == len(gr2[0])
        green = (gr1[0], (gr1[1] + gr2[1]) / 2.0)
        blue = np.histogram(data[1::2][1::2], bins=256, range=[0, max_range])

        colors = 'rgb'
        for n, hist in enumerate((red, green, blue)):
            plt.plot(hist[1][1:], hist[0], colors[n])

        plt.title("%s: %.0f/%.0f/%.0f/%.0f" % info)
        plt.xlabel('intensity')
        plt.ylabel('log')
        plt.yscale('log')
        plt.show()


if __name__ == '__main__':
    main()

