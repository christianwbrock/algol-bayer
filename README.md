#  Capture, display and convert RAW DSLR astronomical images

This package was developed for astro-spectroscopic student projects
at the Dresden Gönnsdorf observatory.

## Installation

    apt-get install gphoto2
    pip install algol-bayer

## Capture and display



    $ bayer_capture_sequence.sh 

        usage: bayer_capture_histograms.sh min-exp-time-s max-exp-time-s
        
               Create and display histograms by doubling the exposure times
               between min and max exposure time.
               Example: bayer_capture_histograms.sh 30 240 will create histograms for exposures of 30, 60, 120 and 240 seconds.


    $ bayer_capture_sequence.sh 

        usage: bayer_capture_sequence.sh object image-count exp-time-s
        
               Capture and image sequence and store them as raw images.
        
               Example: bayer_capture_sequence.sh zetori 20 1800 will capture 20 halve hour exposures zetori_1800_00.cr2, zetori_1800_01.cr2, ...

## Convert

TBD


## Links

 * http://gphoto.org/
 * https://pypi.org/project/rawpy/
 * https://pypi.org/project/algol-reduction/

