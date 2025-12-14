#!/bin/bash

function test_camera()
{
	# We expect two a header lines and a line for each camera
	if  [ 2 = $(gphoto2 --auto-detect | wc -l) ]; then
	  echo No camera found.
	  echo The camera may not be connected or it may have gone to sleep mode?
	  exit 1
	fi
}

function capture_image()
{
  MIRROR_WAIT=2s
  MIRROR_LOCK=$(gphoto2 --list-config | grep -q eosviewfinder && echo eosviewfinder || echo viewfinder)

  SHUTTERSPEED=$1
  FILENAME=$2

  # depending on whether the shutter speed is an integer or a fraction, we set it differently
  if [[ "$SHUTTERSPEED" =~ ^[0-9]+[/.][0-9]+$ ]]; then
    # Note: set -e does not extend inside if statements. That's the reason for the "|| exit 1" below
    gphoto2 --set-config-value=shutterspeed=$SHUTTERSPEED || exit 1
    gphoto2 --quiet \
            --set-config-value=$MIRROR_LOCK=1 \
            --wait-event=$MIRROR_WAIT \
            --capture-image-and-download --filename=$FILENAME || exit 1
  elif [[ "$SHUTTERSPEED" =~ ^[0-9]+$ ]]; then
    gphoto2 --set-config-value=shutterspeed=bulb || exit 1
    gphoto2 --quiet \
            --set-config-value $MIRROR_LOCK=1 \
            --wait-event=$MIRROR_WAIT \
            --bulb $SHUTTERSPEED \
            --capture-image-and-download --filename=$FILENAME || exit 1
  else
    echo "Unrecognized shutter speed \"$SHUTTERSPEED\"."
    echo "Choose one of the following:"
    gphoto2 --get-config=shutterspeed
    exit 1
  fi
}

# test all shutter speeds to fails fast
function test_shutter_speeds()
{
  ALL_EXP_TIMES=$@
  INVALID_EXP_TIMES=()

  for EXP_TIME in $ALL_EXP_TIMES; do
    if [[ "$EXP_TIME" =~ ^[0-9]+$ ]]; then
      # all integers are supported
      continue
    elif [[ "$EXP_TIME" =~ ^[0-9]+[/.][0-9]+$ ]]; then
      # fractions need to be checked
      gphoto2 --set-config-value=shutterspeed=$EXP_TIME || INVALID_EXP_TIMES+=("$EXP_TIME")
    else
      INVALID_EXP_TIMES+=("$EXP_TIME")
    fi
  done

  if [ ${#INVALID_EXP_TIMES[@]} -ne 0 ]; then
    gphoto2 --get-config=shutterspeed
    echo ""
    echo "Unrecognized shutter speed(s): ${INVALID_EXP_TIMES[@]}"
    echo "Choose one of the above or an integer value"
    exit 1
  fi
}

test_camera

# --set-config iso=100
gphoto2 --set-config-value=imageformat=RAW \
	--set-config-value=autoexposuremode=Manual \
	--set-config-value=drivemode=Single \
	--set-config-value=picturestyle=Standard
