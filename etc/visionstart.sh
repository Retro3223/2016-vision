#!/bin/bash

sleep 10s
/usr/bin/python3 /opt/2016-vision/file_streamer.py 2>&1 > /var/log/vision.log 
