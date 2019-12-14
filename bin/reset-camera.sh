#!/bin/bash

function test_camera()
{
	# we expect a header and a line for each camera
	test 2 = $(gphoto2 --auto-detect | wc -l) && echo No camera found && exit 1
}

test_camera


# --set-config iso=100
gphoto2 --set-config imageformat=RAW \
	--set-config autoexposuremode=Manual \
	--set-config drivemode=Single \
	--set-config picturestyle=Standard \
	--set-config shutterspeed=bulb \

