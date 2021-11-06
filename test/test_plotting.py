import os
import sys
import pytest

from bayer.scripts import display_histogram, display_spectrum, visualize_segmentation


def _get_raw_filenames():

    datadir = _find('data')

    filenames = [
        'betori.CR2',
        'alpori.CR2',
        'alpcma.NEF'
    ]

    return [os.path.join(datadir, fn) for fn in filenames]


def _get_fits_filenames():

    datadir = _find('data')

    filenames = [
        'alpleo.FIT',
    ]

    return [os.path.join(datadir, fn) for fn in filenames]


def test_display_raw_histogram():

    for filename in _get_raw_filenames():
        if not os.path.exists(filename):
            continue
        sys.argv = ['dummy', filename]
        display_histogram.main_raw()


def test_display_fits_histogram():

    for filename in _get_fits_filenames():
        if not os.path.exists(filename):
            continue
        sys.argv = ['dummy', filename]
        display_histogram.main_fits()


def test_help_histogram():

    with pytest.raises(SystemExit, match='0'):
        sys.argv = ['dummy', '--help']
        display_histogram.main_raw()


def test_display_raw_spectrum():

    for filename in _get_raw_filenames():
        if not os.path.exists(filename):
            continue
        sys.argv = ['dummy', filename]
        display_spectrum.main_raw()


def test_display_fits_spectrum():

    for filename in _get_fits_filenames():
        if not os.path.exists(filename):
            continue
        sys.argv = ['dummy', filename]
        display_spectrum.main_fits()


def test_help_spectrum():

    with pytest.raises(SystemExit, match='0'):
        sys.argv = ['dummy', '--help']
        display_spectrum.main_raw()


def test_display_raw_contour():

    for filename in _get_raw_filenames():
        if not os.path.exists(filename):
            continue
        sys.argv = ['dummy', filename]
        visualize_segmentation.main_raw()


def test_display_fits_contour():

    for filename in _get_fits_filenames():
        if not os.path.exists(filename):
            continue
        sys.argv = ['dummy', filename]
        visualize_segmentation.main_fits()


def test_help_contour():

    with pytest.raises(SystemExit, match='0'):
        sys.argv = ['dummy', '--help']
        visualize_segmentation.main_raw()


def _find(target):

    here = os.path.dirname(__file__)

    while True:
        join = os.path.join(here, target)
        parent, _ = os.path.split(here)

        if os.path.exists(join):
            return join

        elif parent and parent != here:
            here = parent

        else:
            return None
