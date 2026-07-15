import os
import sys

import numpy as np
from astropy.io import fits

from bayer.scripts import debayer


def test_debayer_with_output(fits_bayer_filename, tmp_path):
    with fits.open(fits_bayer_filename) as hdul:
        input_rows, input_cols = hdul[0].data.shape

    output = tmp_path / 'output-rgb.fits'
    sys.argv = ['dummy', fits_bayer_filename, '--output', str(output)]
    debayer.main()

    with fits.open(output) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        assert header['NAXIS'] == 3
        assert data.shape == (3, input_rows // 2, input_cols // 2)
        assert data.dtype.kind == 'f' and data.dtype.itemsize == 4
        assert header.get('BAYERPAT') is None
        assert header.get('XBAYROFF') is None
        assert header.get('YBAYROFF') is None
        assert np.all(np.isfinite(data))


def test_debayer_wo_output(fits_bayer_filename):
    sys.argv = ['dummy', fits_bayer_filename]
    output = debayer.create_output_filename(fits_bayer_filename)
    try:
        debayer.main()
        assert os.path.exists(output)
    finally:
        if os.path.exists(output):
            os.remove(output)
