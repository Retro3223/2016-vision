#!/bin/bash

# something about structure sensor doesn't work during bootup
# maybe need delay?
sleep 20s
echo "starting vision at $(date)" >> /var/log/vision.log
/usr/bin/python3 /opt/2016-vision/file_streamer.py 1>> /var/log/vision.log 2>&1
