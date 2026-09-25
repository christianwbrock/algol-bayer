import numpy as np
import pytest
from astropy.io import fits

from bayer.utils import fits_white_level


@pytest.mark.parametrize("bitpix, expected", [
    (16, 65535),
    (8, 255),
])
def test_fits_white_level_from_bitpix(bitpix, expected):
    header = fits.Header()
    header['BITPIX'] = bitpix
    assert fits_white_level(header) == expected


@pytest.mark.parametrize("header, data, expected", [
    ({'BITPIX': -32}, [0.0, 1.5], 3),
    ({}, [0.0, 1.0], 1),
    ({}, [0.0, 255.0], 255),
    ({}, [0.0, 256.0], 511),
    ({}, [100.0, 200.0], 255),
])
def test_fits_white_level_from_data(header, data, expected):
    fits_header = fits.Header(header)
    array = np.array(data, dtype=np.float32)
    assert fits_white_level(fits_header, array) == expected


def test_fits_white_level_default_without_bitpix_or_data():
    assert fits_white_level(fits.Header()) == 65535
    assert fits_white_level(None) == 65535


def test_fits_white_level_none_header_estimates_from_data():
    assert fits_white_level(fits.Header(), np.array([2773.0])) == 4095
    assert fits_white_level(None, np.array([2773.0])) == 4095


def test_fits_white_level_from_sample_file(fits_filename):
    with fits.open(fits_filename) as hdul:
        header = hdul[0].header
        data = hdul[0].data
    bitpix = header.get('BITPIX')
    if bitpix is None or int(bitpix) <= 0:
        return
    assert fits_white_level(header, data) == 2 ** int(bitpix) - 1
