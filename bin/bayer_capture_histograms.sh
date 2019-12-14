#!/bin/bash

PROG=$(basename $0)
MIN=$1
MAX=$2

if [ x$MIN = x -o x$MAX = x ]; then
	echo usage: $PROG min-exp-time-s max-exp-time-s
	echo
	echo "       Capture images and display histograms by doubling the exposure times"
	echo "       between min and max exposure time."
	echo "       Example: $PROG 30 240 will create histograms for exposures"
	echo "                of 30, 60, 120 and 240 seconds."
	echo
	exit 1
fi

. $(dirname $0)/reset-camera.sh

function display_hist()
{
	bayer_display_histogram $1
}

gphoto2 --set-config eosviewfinder=1

for ((i=$MIN; i <= $MAX; i*=2)); do
	echo capture $i seconds
	gphoto2 --quiet -B $i --capture-image-and-download --filename=x$i.cr2 && \
	( display_hist x$i.cr2 & )
done

