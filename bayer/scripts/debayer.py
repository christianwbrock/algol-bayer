"""\
For a bayer-masked raw fits file, e.g. generated using an astro color camera and kstars,
convert the raw image into a 3 layer RGB fits image.
"""

import glob
import logging
from argparse import ArgumentParser
from os.path import basename, splitext

import numpy as np
from astropy.io import fits

from bayer.to_rgb import fits_to_layers

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    args = _create_argument_parser().parse_args()

    input_filenames = []
    for pattern in args.filename:
        this = glob.glob(pattern, recursive=True)
        if not this:
            logger.warning(f'no files for pattern "{pattern}"')
        input_filenames += this

    if not input_filenames:
        raise SystemExit(f'no files for any pattern in {args.filename}')

    if len(input_filenames) > 1 and args.output:
        raise SystemExit('cannot use parameter --output with more than one input file')

    output_filenames = [args.output] if args.output else [create_output_filename(fn) for fn in input_filenames]

    for input_filename, output_filename in zip(input_filenames, output_filenames):
        _debayer_fits_file(input_filename, output_filename)


def _debayer_fits_file(input_filename, output_filename):
    with fits.open(input_filename) as hdu_list:
        hdu = hdu_list[0]  # only convert the first
        if len(hdu_list) > 1:
            logger.warning(f'ignore all but the first hdu in "{input_filename}"')

        header = hdu.header

        layers = fits_to_layers(hdu)
        layers = np.asarray(layers, dtype='f')  # float 32
        assert layers.ndim == 3
        assert layers.shape[0] == 3
        # NAXIS*, BITPIX, BZERO and BSCALE will be derived from layers

        for prefix in 'XY':
            name = prefix + 'PIXSZ'
            pixsz = header.get(name)
            if pixsz:
                header.set(name, float(pixsz) * 2,
                           comment=f'{prefix} binned pixel size in microns after de-bayering')

        for prefix in 'XY':
            name = prefix + 'BINNING'
            binning = header.get(name)
            if binning:
                header.set(name, int(binning) * 2, comment=f'{prefix} binning factor after de-bayering')

        for name in 'BAYERPAT', 'XBAYROFF', 'YBAYROFF':
            if header.get(name) is not None:
                header.remove(name)

        header.add_comment(f'De-bayered by {basename(__file__)}; see https://pypi.org/project/algol-bayer/')

        fits.writeto(output_filename, data=layers, header=header)


def create_output_filename(input_filename):
    root, ext = splitext(input_filename)
    return root + '-rgb' + ext


def _create_argument_parser():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('filename', nargs='+', help='one or more fits files containing a single raw color image each')
    parser.add_argument('--output', '-o', nargs='?', help='one fits files containing a single RGB color image')
    return parser
