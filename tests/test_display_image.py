import sys

import numpy as np
import pytest
from astropy.io import fits

from bayer.scripts import display_image
from bayer.to_rgb import fits_to_layers
from bayer.utils import fits_white_level


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


def test_fits_without_scale_uses_bitpix_white_level(tmp_path, monkeypatch):
    data = np.array([[100, 2773], [200, 300]], dtype=np.uint16)
    path = tmp_path / 'dim.fits'
    fits.PrimaryHDU(data).writeto(path)

    with fits.open(path) as hdul:
        expected = fits_white_level(hdul[0].header, fits_to_layers(hdul[0]))
        assert hdul[0].header['BITPIX'] == 16

    captured = {}
    original = display_image.fits_white_level

    def capture_white_level(header, image=None):
        captured['white_level'] = original(header, image)
        captured['bitpix'] = header.get('BITPIX')
        return captured['white_level']

    monkeypatch.setattr(display_image, 'fits_white_level', capture_white_level)
    monkeypatch.setattr(display_image.plt, 'show', lambda: None)
    monkeypatch.setattr(display_image.plt, 'close', lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, 'argv', ['dummy', str(path)])

    display_image.main()

    assert captured['bitpix'] == 16
    assert captured['white_level'] == 65535
    assert captured['white_level'] == expected
    assert captured['white_level'] != 2 ** int(np.ceil(np.log2(2773 + 1))) - 1


def test_fits_skips_empty_primary_and_plots_image_extension(tmp_path, monkeypatch):
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    path = tmp_path / 'empty_primary.fits'
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data)]).writeto(path)

    calls = []

    def capture_plot(filename, image, scale, white_level=None, header=None):
        calls.append(np.asarray(image))

    monkeypatch.setattr(display_image, '_plot_file', capture_plot)
    monkeypatch.setattr(sys, 'argv', ['dummy', str(path)])

    display_image.main()

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [data])


def test_fits_with_only_empty_primary_does_not_plot(tmp_path, monkeypatch, caplog):
    path = tmp_path / 'no_image.fits'
    fits.HDUList([fits.PrimaryHDU()]).writeto(path)

    calls = []
    monkeypatch.setattr(display_image, '_plot_file',
                        lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(sys, 'argv', ['dummy', str(path)])

    display_image.main()

    assert calls == []
    assert 'contains no images' in caplog.text
