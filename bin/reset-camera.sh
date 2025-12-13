#!/bin/bash

function test_camera()
{
	# We expect two a header lines and a line for each camera
	test 2 = $(gphoto2 --auto-detect | wc -l) && echo No camera found && exit 1
}

function capture_image()
{
  MIRROR_WAIT=2s
  MIRROR_LOCK=$(gphoto2 --list-config | grep -q eosviewfinder && echo eosviewfinder || echo viewfinder)

  SHUTTERSPEED=$1
  FILENAME=$2

  # depending on whether the shutterspeed is an integer or a fraction, we set it differently
  if [[ $SHUTTERSPEED == *"/"* ]]; then
    gphoto2 --set-config-value=shutterspeed=$SHUTTERSPEED
    gphoto2 --quiet \
            --set-config-value=$MIRROR_LOCK=1 \
            --wait-event=$MIRROR_WAIT \
            --capture-image-and-download --filename=$FILENAME
  else
    gphoto2 --set-config-value=shutterspeed=bulb
    gphoto2 --quiet \
            --set-config-value $MIRROR_LOCK=1 \
            --wait-event=$MIRROR_WAIT \
            --bulb $SHUTTERSPEED \
            --capture-image-and-download --filename=$FILENAME
  fi
}

test_camera

# --set-config iso=100
gphoto2 --set-config-value=imageformat=RAW \
	--set-config-value=autoexposuremode=Manual \
	--set-config-value=drivemode=Single \
	--set-config-value=picturestyle=Standard
