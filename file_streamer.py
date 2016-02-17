#!/usr/bin/python

import cv2
import argparse
import os, os.path
import time, datetime
import numpy
from vision_processing import Vision
from networktables import NetworkTable

def setup_options_parser():
    parser = argparse.ArgumentParser(description='read structure sensor data.')
    parser.add_argument(
        '--log-dir', dest='log_dir', metavar='LDIR', 
        default='/mnt/',
        help='specify directory in which to log data')
    parser.add_argument('--output-dir', metavar='ODIR', 
        default='/opt/',
        help='specify directory in which to deposit structure.jpg')
    parser.add_argument('--robot', metavar='IP', dest='robot',
        default='roborio-3223-frc',
        help='specify ip address of robot')
    return parser

parser = setup_options_parser()
args = parser.parse_args()

NetworkTable.setIPAddress(args.robot)
NetworkTable.setClientMode()
NetworkTable.initialize()

class DataLogger:
    def __init__(self):
        self.match_running = False
        self.time_start = None
        self.time_stop = None
        self.save_dir = None
        self.sd = NetworkTable.getTable("SmartDashboard")
        self.sd.addTableListener(lambda t, k, v, n: self.value_changed(t,k,v,n))

    def value_changed(self, table, key, value, is_new):
        if key == "auto_on" and not self.match_running:
            self.match_running = True
            self.save_dir = os.path.join(args.log_dir, 
                datetime.datetime.now().isoformat())
            try: 
                os.makedirs(self.save_dir)
            except:
                # forget that idea, i guess
                self.match_running = False
            self.time_start = time.time()
            self.time_stop = self.time_start + 150

    def stop_when_done(self):
        now = time.time()
        if self.match_running and now > self.time_stop:
            self.match_running = False

    def log_data(self, depth, ir):
        if self.match_running:
            try:
                now_milis = int(time.time() * 1000)
                fnom = os.path.join('/mnt', self.save_dir, str(now_milis))
                numpy.savez_compressed(fnom, {'depth': depth, 'ir': ir})
            except:
                # don't stop the main loop!
                pass
        self.stop_when_done()
            

file_name = os.path.join(args.output_dir, "structure.jpg")
tmp_name = os.path.join(args.output_dir, "structure.tmp.jpg")
vision = Vision()
vision.mode = 100

logger = DataLogger()

now = time.time()

with vision:
    while True:
        vision.get_depths()
        vision.idepth_stats()
        vision.set_display()
        cv2.imwrite(tmp_name, vision.display)
        os.rename(tmp_name, file_name)
        logger.log_data(vision.depth, vision.ir)
        time.sleep(0.05)
