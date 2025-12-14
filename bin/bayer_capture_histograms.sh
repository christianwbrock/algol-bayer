#!/bin/bash

set -eu

PROG=$(basename $0)
ALL_EXP_TIMES=$@

if [ -z "$ALL_EXP_TIMES" ]; then
	echo usage: $PROG exposure-time-1 max-exp-time-2 ...
	echo
	echo "       Capture images and display histograms for a list of exposure times."

	echo "       Example: $PROG 1/1000 1/100 1/10 1 10 100 will create histograms for exposures"
	echo "                of 1/1000, 1/100, 1/10, 1, 10 and 100 seconds."
	echo
	exit 1
fi

. $(dirname $0)/reset-camera.sh


test_shutter_speeds $ALL_EXP_TIMES

for EXP_TIME in $ALL_EXP_TIMES; do
  echo capture $EXP_TIME seconds

  # for display purpose remove slash from exposure time using tr
  DISPLAY_TIME=$(echo $EXP_TIME | tr '/' '_')
  IMAGE_NAME=hist_$DISPLAY_TIME.cr2

	capture_image $EXP_TIME $IMAGE_NAME && \
	(bayer_display_histogram $IMAGE_NAME &)
done
