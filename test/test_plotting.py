import os
import sys
import pytest

from bayer.scripts import display_histogram, display_spectrum, show_rgb_layers


def get_filenames():

    datadir = _find('data')

    filenames = [
        '2015-02-20-betori/orig/IMG_2375.CR2',
        '2015-02-19-alpori/orig/IMG_2236.CR2',
        '2014-03-12-winter6/orig/DSC_0828_Sirius.NEF'
    ]

    return [os.path.join(datadir, fn) for fn in filenames]


def test_display_histogram():

    for filename in get_filenames():
        if not os.path.exists(filename):
            continue
        sys.argv = ['dummy', filename]
        display_histogram.main()


def test_help_histogram():

    with pytest.raises(SystemExit, match='0'):
        sys.argv = ['dummy', '--help']
        display_histogram.main()


def test_display_spectrum():

    for filename in get_filenames():
        if not os.path.exists(filename):
            continue
        sys.argv = ['dummy', filename]
        display_spectrum.main()


def test_help_spectrum():

    with pytest.raises(SystemExit, match='0'):
        sys.argv = ['dummy', '--help']
        display_spectrum.main()


def test_display_contour():

    for filename in get_filenames():
        if not os.path.exists(filename):
            continue
        sys.argv = ['dummy', filename]
        show_rgb_layers.main()


def test_help_contour():

    with pytest.raises(SystemExit, match='0'):
        sys.argv = ['dummy', '--help']
        show_rgb_layers.main()


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
