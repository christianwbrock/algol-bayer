# Capture, display and convert RAW DSLR astronomical images

This package was developed for astro-spectroscopic student projects
at the Dresden Gönnsdorf observatory.

There are method to

* [Capture DSLR raw images](#capture-dslr-raw-images)
* [Display raw DSLR images and spectra](#display-raw-dslr-images-and-spectra)
* [Display fits images and spectra](#display-fits-images-and-spectra)
* [Dark and flat correction](#dark-and-flat-correction)
* [Debayer a 2d into a 3d fits file](#debayer-a-2d-into-a-3d-fits-file)

## Installation

    apt-get install gphoto2
    pip install algol-bayer

## Capture DSLR raw images

### Capture and display as histograms

Capture a sequence of DSLR raw images using increasing exposure times
and [display the images as histogram](#display-raw-dslr-image-as-histogram).

    bayer_capture_histograms.sh

### Capture and display as spectra

Capture a sequence of DSLR raw images and try to
[display them as spectra](#display-raw-dslr-image-as-spectrum).

    bayer_capture_spectra.sh

### Capture and display as image

Capture a sequence of DSLR raw images and [display them as RGB images](#display-raw-dslr-image).

    bayer_capture_images.sh

## Display raw DSLR images and spectra

### Display raw DSLR image

![alt text](./docs/article/img/alpori.png)

    bayer_display_image --help

### Display raw DSLR image as histogram

![alt text](./docs/article/img/alpori_hist.png)

    bayer_display_histogram --help

### Display raw DSLR image as spectrum

There is a published paper explaining how this works.
It can be found online or in the `docs/article` folder.

![alt text](./docs/article/img/alpori_spec.png)

    bayer_display_spectrum --help

## Display fits images and spectra

### Display fits image as histogram

![alt text](./docs/article/img/alpleo_hist.png)

    fits_display_histogram --help

### Display fits image as spectrum

![alt text](./docs/article/img/alpleo_spec.png)

There is a published paper explaining how this works.
It can be found online or in the `docs/article` folder.

    fits_display_spectrum --help

## Dark and flat correction

An example of how to do this can be found in `tests/test_darkflat.py`

### Create master dark

... or a master-flat-dark from a set of dark images.

    fits_create_master_dark --help

### Create master flat

... from a master-flat-dark and a set of flat images.

    fits_create_master_flat --help

### Apply master dark and master flat to light images

    fits_apply_darks_and_flats --help

## Conversions

### Debayer a 2d into a 3d fits file

    fits_debayer --help

## Links

* https://github.com/christianwbrock/algol-bayer
* http://gphoto.org/
* https://pypi.org/project/rawpy/

