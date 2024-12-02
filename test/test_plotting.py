import os
import sys
import pytest

from bayer.scripts import display_histogram, display_spectrum, visualize_segmentation


def test_display_raw_histogram(raw_filename):

    if not os.path.exists(raw_filename):
        pytest.skip()

    sys.argv = ['dummy', raw_filename]
    display_histogram.main_raw()


def test_display_fits_histogram(fits_filename):
    if not os.path.exists(fits_filename):
        pytest.skip()
    sys.argv = ['dummy', fits_filename]
    display_histogram.main_fits()


def test_help_histogram():

    with pytest.raises(SystemExit, match='0'):
        sys.argv = ['dummy', '--help']
        display_histogram.main_raw()


def test_display_raw_spectrum(raw_filename):

    if not os.path.exists(raw_filename):
        pytest.skip()
    sys.argv = ['dummy', raw_filename]
    display_spectrum.main_raw()


def test_display_fits_spectrum(fits_filename):

    if not os.path.exists(fits_filename):
        pytest.skip()
    sys.argv = ['dummy', fits_filename]
    display_spectrum.main_fits()


def test_help_spectrum():

    with pytest.raises(SystemExit, match='0'):
        sys.argv = ['dummy', '--help']
        display_spectrum.main_raw()


def test_display_raw_contour(raw_filename):

    if not os.path.exists(raw_filename):
        pytest.skip()
    sys.argv = ['dummy', raw_filename]
    visualize_segmentation.main_raw()


def test_display_fits_contour(fits_filename):

    if not os.path.exists(fits_filename):
        pytest.skip()
    sys.argv = ['dummy', fits_filename]
    visualize_segmentation.main_fits()


def test_help_contour():

    with pytest.raises(SystemExit, match='0'):
        sys.argv = ['dummy', '--help']
        visualize_segmentation.main_raw()
