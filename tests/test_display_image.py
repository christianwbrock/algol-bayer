import sys

import numpy as np
import pytest

from bayer.scripts import display_image


def test_display_image(image_filename):

    sys.argv = ['dummy', '--scale', image_filename]
    display_image.main()


def test_main_displays_each_raw_file_once(monkeypatch):
    calls = []

    class FakeRaw:
        white_level = 4095

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr('bayer.scripts.display_image.multi_glob',
                        lambda patterns: ['a.cr2', 'b.cr2', 'c.cr2'])
    monkeypatch.setattr('rawpy.imread', lambda fn: FakeRaw())
    monkeypatch.setattr('bayer.scripts.display_image.rawpy_to_rgb', lambda raw: np.ones((3, 2, 2)))
    monkeypatch.setattr('bayer.scripts.display_image._plot_file',
                        lambda fn, *args: calls.append(fn))

    sys.argv = ['dummy', 'ignored']
    display_image.main()

    assert calls == ['a.cr2', 'b.cr2', 'c.cr2']


def test_help():

    with pytest.raises(SystemExit, match='0'):
        sys.argv = ['dummy', '--help']
        display_image.main()
