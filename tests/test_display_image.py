import sys

import pytest

from bayer.scripts import display_image


def test_display_image(image_filename):

    sys.argv = ['dummy', '--scale', image_filename]
    display_image.main()


def test_help():

    with pytest.raises(SystemExit, match='0'):
        sys.argv = ['dummy', '--help']
        display_image.main()
