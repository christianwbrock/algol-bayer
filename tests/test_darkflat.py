import sys

import numpy as np
from astropy.io import fits

from bayer.scripts import darkflat


def _write_mono_fits(path, data, **extra_headers):
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=np.float32))
    hdu.header['NAXIS'] = 2
    hdu.header['NAXIS1'] = data.shape[1]
    hdu.header['NAXIS2'] = data.shape[0]
    for key, value in extra_headers.items():
        hdu.header[key] = value
    hdu.writeto(path, overwrite=True)


def test_create_output_filename():
    assert darkflat.create_output_filename('/foo/bar.fits', '/out/', '-d') == '/out/bar-d.fits'


def test_create_output_filename_without_slash():
    assert darkflat.create_output_filename('/foo/bar.fits', '/out', '-d') == '/out/bar-d.fits'


def test_create_master_dark_mean(tmp_path):
    ones = np.ones((2, 2), dtype=np.float32)
    for i, scale in enumerate([1, 2, 6]):
        _write_mono_fits(tmp_path / f'd{i}.fits', ones * scale)
    out = tmp_path / 'md.fits'
    sys.argv = ['dummy', str(tmp_path / 'd*.fits'), '-o', str(out), '--algorithm', 'mean']
    darkflat.create_master_dark()
    assert out.exists()
    with fits.open(out) as hdul:
        np.testing.assert_allclose(ones * 3, hdul[0].data)


def test_create_master_dark_median(tmp_path):
    ones = np.ones((2, 2), dtype=np.float32)
    for i, scale in enumerate([1, 2, 6]):
        _write_mono_fits(tmp_path / f'd{i}.fits', ones * scale)
    out = tmp_path / 'md.fits'
    sys.argv = ['dummy', str(tmp_path / 'd*.fits'), '-o', str(out), '--algorithm', 'median']
    darkflat.create_master_dark()
    assert out.exists()
    with fits.open(out) as hdul:
        np.testing.assert_allclose(ones * 2, hdul[0].data)


def test_create_master_dark_sigma3(tmp_path):
    ones = np.ones((2, 2), dtype=np.float32)
    for i, scale in enumerate([0, 0, 0, 100, -100]):
        _write_mono_fits(tmp_path / f'd{i}.fits', ones * float(scale))
    out = tmp_path / 'md.fits'
    sys.argv = ['dummy', str(tmp_path / 'd*.fits'), '-o', str(out), '--algorithm', 'sigma3']
    darkflat.create_master_dark()
    assert out.exists()
    with fits.open(out) as hdul:
        np.testing.assert_allclose(np.zeros((2, 2), dtype=np.float32), hdul[0].data)


def test_apply_master_dark_only(tmp_path):
    light = np.ones((4, 4), dtype=np.float32) * 100.0
    dk = np.ones((4, 4), dtype=np.float32) * 10.0
    _write_mono_fits(tmp_path / 'light.fits', light)
    _write_mono_fits(tmp_path / 'md.fits', dk)
    outdir = tmp_path / 'out'
    outdir.mkdir()
    sys.argv = [
        'dummy', str(tmp_path / 'light.fits'),
        '--master-dark', str(tmp_path / 'md.fits'),
        '-o', str(outdir),
    ]
    darkflat.apply_darks_and_flats()
    outs = list(outdir.glob('*.fits'))
    assert len(outs) == 1
    with fits.open(outs[0]) as hdul:
        np.testing.assert_allclose(hdul[0].data, 90.0)


def test_apply_master_dark_and_master_flat(tmp_path):
    light = np.ones((4, 4), dtype=np.float32) * 100.0
    flat = np.ones((4, 4), dtype=np.float32) * 2.0
    dark = np.ones((4, 4), dtype=np.float32) * 10.0
    _write_mono_fits(tmp_path / 'light.fits', light)
    _write_mono_fits(tmp_path / 'md.fits', dark)
    _write_mono_fits(tmp_path / 'mf.fits', flat)
    outdir = tmp_path / 'out'
    outdir.mkdir()
    sys.argv = [
        'dummy', str(tmp_path / 'light.fits'),
        '--master-dark', str(tmp_path / 'md.fits'),
        '--master-flat', str(tmp_path / 'mf.fits'),
        '-o', str(outdir),
    ]
    darkflat.apply_darks_and_flats()
    outs = list(outdir.glob('*.fits'))
    assert len(outs) == 1
    with fits.open(outs[0]) as hdul:
        np.testing.assert_allclose(hdul[0].data, 90.0 / 2.0)


def test_create_master_flat_non_bayer(tmp_path):
    flat = np.full((4, 4), 100.0, dtype=np.float32)
    fd = np.full((4, 4), 5.0, dtype=np.float32)
    _write_mono_fits(tmp_path / 'flat0.fits', flat)
    _write_mono_fits(tmp_path / 'flat1.fits', flat)
    _write_mono_fits(tmp_path / 'fd.fits', fd)
    out = tmp_path / 'mf.fits'
    sys.argv = [
        'dummy', str(tmp_path / 'flat*.fits'),
        '--master-flat-dark', str(tmp_path / 'fd.fits'),
        '-o', str(out),
        '--algorithm', 'mean',
    ]
    darkflat.create_master_flat()
    assert out.exists()
    with fits.open(out) as hdul:
        np.testing.assert_allclose(hdul[0].data, 1.0)


def test_create_master_flat_bayer(tmp_path):
    """BAYERPAT triggers per-Bayer-quadrant normalization in _normalize_flat."""
    flat = np.full((4, 4), 100.0, dtype=np.float32)
    fd = np.full((4, 4), 5.0, dtype=np.float32)
    _write_mono_fits(tmp_path / 'flat0.fits', flat, BAYERPAT='RGGB')
    _write_mono_fits(tmp_path / 'flat1.fits', flat, BAYERPAT='RGGB')
    _write_mono_fits(tmp_path / 'fd.fits', fd)
    out = tmp_path / 'mf-bayer.fits'
    sys.argv = [
        'dummy', str(tmp_path / 'flat*.fits'),
        '--master-flat-dark', str(tmp_path / 'fd.fits'),
        '-o', str(out),
        '--algorithm', 'mean',
    ]
    darkflat.create_master_flat()
    assert out.exists()
    with fits.open(out) as hdul:
        np.testing.assert_allclose(hdul[0].data, 1.0)
