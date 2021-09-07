import os.path
from argparse import ArgumentParser

import matplotlib.cm
import numpy as np
import rawpy
from matplotlib import pyplot as plt

from bayer.extraction import FastExtraction
from bayer.to_rgb import rawpy_to_rgb


def main():
    parser = ArgumentParser(description='Display histogram of a raw image')
    parser.add_argument('filename', nargs='+', help='one or more raw files containing bayer matrices')
    parser.add_argument('--sigma', '-s', default=3.0, help='sigma used for clipping')
    parser.add_argument('--clipping', '-c', default=10.0, help='clip background at mean + clipping * stddev')

    args = parser.parse_args()

    for filename in args.filename:
        if not os.path.exists(filename):
            continue

        with rawpy.imread(filename) as raw:
            layers = rawpy_to_rgb(raw)

        fast = FastExtraction(rgb_layers=layers, sigma=args.sigma, clipping=args.clipping)

        three_sigma = fast.background_mean + fast.background_stddev * 3
        ten_sigma = fast.background_mean + fast.background_stddev * 10
        threshold = np.nanmean(fast.clipped_rgb, axis=(1, 2))

        levels = np.array([three_sigma, ten_sigma, threshold]).T
        contour_colors = 'green red blue'.split()
        contour_lw = [1, 2, 2]
        layer_colors = 'red green blue'.split()
        labels = [[f"${a:.0f} = {args.sigma} \\sigma$",
                   f'${b:.0f} = {args.clipping} \\sigma$',
                   f'${c:.0f} = mean > {args.clipping} \\sigma$'] for a, b, c in levels]

        layer_count, row_count, column_count = fast.rgb.shape

        fig = plt.figure(figsize=(6, 12))

        for idx_layer in range(layer_count):
            ax = fig.add_subplot(layer_count, 1, idx_layer + 1)
            ax.imshow(fast.rgb[idx_layer], cmap=matplotlib.cm.gray)
            cs = ax.contour(np.arange(column_count), np.arange(row_count), fast.rgb[idx_layer],
                            levels=levels[idx_layer], colors=contour_colors)
            for idx_label in range(len(labels[idx_layer])):
                label = labels[idx_layer][idx_label]
                cs.collections[idx_label].set_label(label)
                cs.collections[idx_label].set_lw(contour_lw[idx_label])
            ax.set_title(layer_colors[idx_layer])
            ax.legend()

        fig.canvas.set_window_title(os.path.basename(filename))
        fig.tight_layout()
        plt.show()
        plt.close(fig)
