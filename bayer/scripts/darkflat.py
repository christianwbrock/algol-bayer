'''\
Apply dork and flat images to one or more light images.
'''
import glob
import sys
from os.path import basename, splitext, split
from argparse import ArgumentParser

import numpy as np
from astropy.io import fits

import logging

DEFAULT_INFIX_DF = '-df'
DEFAULT_INFIX_D = '-d'

program_name = sys.argv[0] if sys.argv and sys.argv[0] else basename(__file__)
logger = logging.getLogger(program_name)


def apply_darks_and_flats():
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(description="""\
        Apply a master-dark and optionally also a master-flat to a list of fits file.""")

    parser.add_argument('lights', nargs='+')
    parser.add_argument('--master-dark', required=True)
    parser.add_argument('--master-flat', required=False)
    parser.add_argument('--output-folder', '-o', default='.', help='default=%(default)s')
    parser.add_argument('--infix', help=f'defaults to either {DEFAULT_INFIX_D} or {DEFAULT_INFIX_DF}')
    parser.add_argument('--output-format', choices=['f4', 'u2', 'auto'], default='auto', help='default=%(default)s')
    parser.add_argument('--overwrite', default=False, action='store_true', help='Allow overwriting output')
    args = parser.parse_args()

    lights = _load_from_pattern(args.lights)
    input_filenames = _filenames(args.lights)

    master_dark = _load_from_pattern(args.master_dark)
    assert master_dark.shape[0] == 1
    output = lights - master_dark

    if args.master_flat:
        master_flat = _load_from_pattern(args.master_flat)
        assert master_flat.shape[0] == 1
        output = output / master_flat
        infix = args.infix or DEFAULT_INFIX_DF
    else:
        infix = args.infix or DEFAULT_INFIX_D

    output = _format_array(output, args.output_format)

    output_filenames = [create_output_filename(fn, args.output_folder, infix) for fn in input_filenames]
    for o, inp, out in zip([output[i] for i in range(output.shape[0])], input_filenames, output_filenames):
        _write_fits_using_header(o, inp, out, args.overwrite)


def create_master_dark():
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(description="""\
        Create a master dark, flat-dark or bias from a list of files.""")

    parser.add_argument('darks', nargs='+')
    parser.add_argument('--algorithm', choices=['mean', 'median', 'sigma3'], default='sigma3',
                        help='default=%(default)s')
    parser.add_argument('--output', '-o', default='master-dark.fits', help='default=%(default)s')
    parser.add_argument('--output-format', choices=['f4', 'u2', 'auto'], default='auto', help='default=%(default)s')
    parser.add_argument('--overwrite', default=False, action='store_true', help='Allow overwriting output')
    args = parser.parse_args()

    darks = _load_from_pattern(args.darks)
    output = _average(darks, args.algorithm)
    output = _format_array(output, args.output_format)

    _write_fits_using_header(output, _filenames(args.darks)[0], args.output, args.overwrite)


def create_master_flat():
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(description="""\
        Create a master flat from a list of flats and a master dark.""")

    parser.add_argument('flats', nargs='+')
    parser.add_argument('--algorithm', choices=['mean', 'median', 'sigma3'], default='sigma3',
                        help='default=%(default)s')
    parser.add_argument('--master-flat-dark', required=True)
    parser.add_argument('--output', '-o', default='./master-flat.fits', help='default=%(default)s')
    parser.add_argument('--output-format', choices=['f4', 'auto'], default='auto', help='default=%(default)s')
    parser.add_argument('--overwrite', default=False, action='store_true', help='Allow overwriting output')
    args = parser.parse_args()

    first_filename = _filenames(args.flats)[0]
    flats = _load_from_pattern(args.flats)
    master_dark = _load_from_pattern(args.master_flat_dark)
    assert master_dark.shape[0] == 1
    flats = flats - master_dark
    output = _average(flats, args.algorithm)
    output = _normalize_flat(output, first_filename)
    output = _format_array(output, args.output_format)

    _write_fits_using_header(output, first_filename, args.output, args.overwrite)


def _normalize_flat(flat, filename):
    """Praxis has shown, that flats are never white. For RGB flats we want to normalize by channel."""
    assert flat.ndim == 2
    with fits.open(filename) as hdu_list:
        header = hdu_list[0].header
        is_bayer = header.get('BAYERPAT') is not None
        if is_bayer:
            assert header.get('NAXIS') == 2
            # ignore the last odd row or column
            rows, columns = flat.shape
            rows = rows // 2 * 2
            columns = columns // 2 * 2
            for row, column in (0, 0), (0, 1), (1, 0), (1, 1):
                mean = np.nanmean(flat[row:rows:2, column:columns:2])
                flat[row:rows:2, column:columns:2] = flat[row:rows:2, column:columns:2] / mean
        else:
            rows, columns = flat.shape
            rows = rows // 2 * 2
            columns = columns // 2 * 2
            flat = flat / np.nanmean(flat[0:rows, 0:columns])
        return flat


def _format_array(output, output_format):
    if output_format == 'auto':
        # convert double to single float
        if output.dtype == np.float64:
            output = np.asarray(output, dtype=np.float32)
    else:
        output = np.asarray(output, dtype=output_format)
    return output


def _write_fits_using_header(data, input_filename, output_filename, overwrite):
    with fits.open(input_filename) as hdu_list:
        header = hdu_list[0].header
        header.add_comment(f'darks and/or flats by {program_name}; see https://pypi.org/project/algol-bayer/')
        fits.writeto(output_filename, data, header=header, overwrite=overwrite)


def _average(input, algorithm):
    assert input.ndim == 3
    if algorithm == 'mean':
        output = np.nanmedian(input, axis=0)
    elif algorithm == 'median':
        output = np.nanmedian(input, axis=0)
    else:
        assert algorithm == 'sigma3'
        clipped = input
        for i in range(3):
            mean = np.nanmean(clipped, axis=0)
            stddev = np.nanstd(clipped, axis=0)
            clipped = np.where(np.logical_and(mean - 3 * stddev <= input, input <= mean + 3 * stddev), input, np.nan)

        output = np.nanmean(clipped, axis=0)

    assert output.ndim == 2
    return output


def _filenames(patterns):
    if isinstance(patterns, str):
        patterns = [patterns]

    result = []
    for pattern in patterns:
        filesnames = glob.glob(pattern, recursive=True)
        if not filesnames:
            logger.warning(f'pattern {pattern} yields no files')
        result += filesnames
    return result


def _load_from_pattern(pattern):
    filenames = _filenames(pattern)
    if not filenames:
        raise SystemExit(f'no files for pattern "{pattern}"')
    # TODO load more than one image per file?
    return np.asarray([fits.open(fn)[0].data for fn in filenames])


def create_output_filename(input_filename, folder, infix):
    if folder[-1] != '/':
        folder += '/'
    __, name = split(input_filename)
    root, ext = splitext(folder + name)
    return root + infix + ext
