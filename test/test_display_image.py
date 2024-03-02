import os
import sys

import pytest

from bayer.scripts import display_image


def _get_filenames():

    data_dir = _find('data')

    filenames = [
        'alpcma.NEF',
        'alpleo.FIT',
        'alpori.CR2',
        'betori.CR2',
        'minus20degC_Dark_60_secs_100.fits',
    ]

    return [os.path.join(data_dir, fn) for fn in filenames]


def test_display_image():

    for filename in _get_filenames():
        if not os.path.exists(filename):
            continue
        sys.argv = ['dummy', '--scale', filename]
        display_image.main()


def test_help():

    with pytest.raises(SystemExit, match='0'):
        sys.argv = ['dummy', '--help']
        display_image.main()


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
