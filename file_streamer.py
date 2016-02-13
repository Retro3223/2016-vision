#!/usr/bin/python

import cv2
import os
import time
from vision_processing import Vision

file_name = "/opt/structure.jpg"
tmp_name = "/opt/structure.tmp.jpg"
vision = Vision()
vision.mode = 100
with vision:
    while True:
        vision.get_depths()
        vision.idepth_stats()
        vision.set_display()
        cv2.imwrite(tmp_name, vision.display)
        os.rename(tmp_name, file_name)
        time.sleep(0.15)
