#!/bin/bash

# something about structure sensor doesn't work during bootup
# maybe need delay?
sleep 20s
/usr/bin/python3 /opt/2016-vision/file_streamer.py 2>&1 > /var/log/vision.log 
