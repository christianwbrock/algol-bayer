import os
import sys

import pytest

from bayer.scripts import display_histogram, display_spectrum, visualize_segmentation


def test_display_raw_histogram(raw_filename):
    _test_method(raw_filename, display_histogram.main_raw)


def test_help_raw_histogram():
    _test_help(display_histogram.main_raw)


def test_display_fits_histogram(fits_filename):
    _test_method(fits_filename, display_histogram.main_fits)


def test_help_fits_histogram():
    _test_help(display_histogram.main_fits)


def test_display_raw_spectrum(raw_filename):
    _test_method(raw_filename, display_spectrum.main_raw)


def test_help_raw_spectrum():
    _test_help(display_spectrum.main_raw)


def test_display_fits_spectrum(fits_filename):
    _test_method(fits_filename, display_spectrum.main_fits)


def test_help_fits_spectrum():
    _test_help(display_spectrum.main_fits)


def test_display_raw_contour(raw_filename):
    _test_method(raw_filename, visualize_segmentation.main_raw)


def test_help_raw_contour():
    _test_help(visualize_segmentation.main_raw)


def test_display_fits_contour(fits_filename):
    _test_method(fits_filename, visualize_segmentation.main_fits)


def test_help_fits_contour():
    _test_help(visualize_segmentation.main_fits)


def _test_method(filename, method):
    if not os.path.exists(filename):
        pytest.skip()

    sys.argv = ['dummy', filename]
    method()


def _test_help(method):
    with pytest.raises(SystemExit, match='0'):
        sys.argv = ['dummy', '--help']
        method()
