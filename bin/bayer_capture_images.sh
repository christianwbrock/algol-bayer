#!/bin/bash

PROG=$(basename $0)
OBJ=$1
COUNT=$2
EXP_TIME=$3

if [ x$OBJ = x -o x$EXP_TIME = x -o x$COUNT = x ]; then
	echo usage: $PROG object image-count exp-time-s
	echo
	echo "       Capture an image sequence, store and display them as raw images."
	echo ""
	echo "       Examples:"
	echo "           $PROG zetori 20 1800"
	echo "                will capture 20 halve hour exposures"
	echo "                zetori_1800_00.cr2, zetori_1800_01.cr2, ..."
	echo ""
	echo "           $PROG alpori 10 1/4"
	echo "                will capture 10 quater second exposures"
	echo "                alpori_1_4_00.cr2, alpori_1_4_01.cr2, ..."
	echo ""
	exit 1
fi

. $(dirname $0)/reset-camera.sh

function preview()
{
	bayer_display_image $1
}


for (( i=0; i < $COUNT; ++i)); do
	DISPLAY_TIME=$(echo $EXP_TIME | tr '/' '_')
	printf -v IMAGE_NAME "%s_%s_%02d.cr2" $OBJ $DISPLAY_TIME $i
	echo $(($i+1))/$COUNT capture and download $IMAGE_NAME

	capture_image $EXP_TIME $IMAGE_NAME && \
	( preview $IMAGE_NAME & )
done
