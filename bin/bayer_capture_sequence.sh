#!/bin/bash

PROG=$(basename $0)
OBJ=$1
COUNT=$2
TIME=$3

if [ x$OBJ = x -o x$TIME = x -o x$COUNT = x ]; then
	echo usage: $PROG object image-count exp-time-s
	echo
	echo "       Capture an image sequence, store them as raw images and"
	echo "       try to display them as spectra."
	echo ""
	echo "       Example: $PROG zetori 20 1800 will capture 20 halve hour exposures"
	echo "                zetori_1800_00.cr2, zetori_1800_01.cr2, ..."
	echo ""
	exit 1
fi

. $(dirname $0)/reset-camera.sh

gphoto2 --set-config eosviewfinder=1

function preview()
{
	bayer_display_spectrum --dont-cut $1
}


for (( i=0; i < $COUNT; ++i)); do
	printf -v image "%s_%s_%02d.cr2" $OBJ $TIME $i
	echo $(($i+1))/$COUNT capture and download $image
	gphoto2 --quiet -B $TIME --filename $image --capture-image-and-download && \
	( preview $image & )
done
