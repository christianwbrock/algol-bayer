import os
import sys

import numpy as np
import pytest
from astropy.io import fits

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


def test_fits_spectrum_skips_empty_primary_and_extracts_extension(tmp_path, monkeypatch):
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    path = tmp_path / 'empty_primary.fits'
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data)]).writeto(path)

    plotted = []

    class DummyExtractor:
        def __init__(self, image_layers, sigma=3.0, clipping=10.0):
            self.image_layers = image_layers

    monkeypatch.setattr(display_spectrum, 'FastExtraction', DummyExtractor)
    monkeypatch.setattr(display_spectrum, '_plot_file',
                        lambda filename, extractor, white_level, store:
                        plotted.append(np.asarray(extractor.image_layers)))
    monkeypatch.setattr(sys, 'argv', ['dummy', str(path)])

    display_spectrum.main_fits()

    assert len(plotted) == 1
    np.testing.assert_array_equal(plotted[0], [data])


def test_fits_spectrum_with_only_empty_primary_does_not_plot(tmp_path, monkeypatch, caplog):
    path = tmp_path / 'no_image.fits'
    fits.HDUList([fits.PrimaryHDU()]).writeto(path)

    plotted = []
    monkeypatch.setattr(display_spectrum, '_plot_file',
                        lambda *args, **kwargs: plotted.append(args))
    monkeypatch.setattr(sys, 'argv', ['dummy', str(path)])

    display_spectrum.main_fits()

    assert plotted == []
    assert 'contains no images' in caplog.text


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
