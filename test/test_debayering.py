import os
import sys
import tempfile

from bayer.scripts import debayer


def test_debayer_with_output():
    for fn in _get_filenames():
        output = tempfile.mktemp(suffix='-rgb.fits')
        sys.argv = ['dummy', fn, "--output", output]
        try:
            debayer.main()
        finally:
            os.remove(output)


def test_debayer_wo_output():
    for fn in _get_filenames():
        sys.argv = ['dummy', fn]
        output = debayer.create_output_filename(fn)
        try:
            debayer.main()
        finally:
            os.remove(output)


def _get_filenames():

    data_dir = _find('data')

    filenames = [
        'minus20degC_Dark_60_secs_100.fits',
    ]

    return [os.path.join(data_dir, fn) for fn in filenames]


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
