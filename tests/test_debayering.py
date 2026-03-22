import os
import sys
import tempfile

from bayer.scripts import debayer


def test_debayer_with_output(fits_bayer_filename):
    output = tempfile.mktemp(suffix='-rgb.fits')
    sys.argv = ['dummy', fits_bayer_filename, "--output", output]
    try:
        debayer.main()
    finally:
        os.remove(output)


def test_debayer_wo_output(fits_bayer_filename):
    sys.argv = ['dummy', fits_bayer_filename]
    output = debayer.create_output_filename(fits_bayer_filename)
    try:
        debayer.main()
    finally:
        os.remove(output)
