import glob
import os
import sys
import tempfile

import pytest

from bayer.scripts import darkflat


def test_flats_and_darks():
    data_folder = '/home/guest/Documents/Astro/data/2022-01-08-M1'
    if not os.path.exists(data_folder):
        pytest.skip("Data folder is not available")

    output_folder = tempfile.mkdtemp()
    try:
        lights = data_folder + '/Light/*'
        darks = data_folder + '/Dark/*'
        flats = data_folder + '/Flat/*'
        flat_darks = data_folder + '/FlatDark/*'

        master_dark = output_folder + '/master-dark.fits'
        sys.argv = ['dummy', darks, '-o', master_dark]
        darkflat.create_master_dark()

        master_flat_dark = output_folder + '/master-flat-dark.fits'
        sys.argv = ['dummy', flat_darks, '-o', master_flat_dark]
        darkflat.create_master_dark()

        master_flat = output_folder + '/master-flat.fits'
        sys.argv = ['dummy', flats, '--master-flat-dark', master_flat_dark, '-o', master_flat]
        darkflat.create_master_flat()

        sys.argv = ['dummy', "--output-folder", output_folder, lights,
                    '--master-dark', master_dark]
        darkflat.apply_darks_and_flats()

        sys.argv = ['dummy', "--output-folder", output_folder, lights,
                    '--master-dark', master_dark, '--master-flat', master_flat]
        darkflat.apply_darks_and_flats()

    finally:
        for fn in glob.glob(output_folder + '/*'):
            os.remove(fn)
        os.rmdir(output_folder)
