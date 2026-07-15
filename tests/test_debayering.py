import os
import sys

from bayer.scripts import debayer


def test_debayer_with_output(fits_bayer_filename, tmp_path):
    output = tmp_path / 'output-rgb.fits'
    sys.argv = ['dummy', fits_bayer_filename, '--output', str(output)]
    debayer.main()
    assert output.exists()


def test_debayer_wo_output(fits_bayer_filename):
    sys.argv = ['dummy', fits_bayer_filename]
    output = debayer.create_output_filename(fits_bayer_filename)
    try:
        debayer.main()
    finally:
        if os.path.exists(output):
            os.remove(output)
