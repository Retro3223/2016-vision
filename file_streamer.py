#!/usr/bin/python

import cv2
import argparse
import os
import os.path
import time
from vision_processing import Vision
from networktables import NetworkTable
from data_logger import DataLogger

def find_roborio():
    possible_hosts = ["roborio-3223-frc.local","roborio-3223-frc","roborio-3223-frc.frc-robot.local","roborio-3223-frc.frc-robot"]
    for host in possible_hosts:
        response = os.system("ping -c 1 " + host)
        if response == 0:
            return host
    return possible_hosts[0]

def setup_options_parser():
    parser = argparse.ArgumentParser(description='read structure sensor data.')
    parser.add_argument(
        '--log-dir', dest='log_dir', metavar='LDIR',
        default='/opt/data',
        help='specify directory in which to log data')
    parser.add_argument(
        '--output-dir', metavar='ODIR',
        default='/opt/',
        help='specify directory in which to deposit structure.jpg')
    return parser

parser = setup_options_parser()
args = parser.parse_args()

NetworkTable.setIPAddress("10.32.23.2")
NetworkTable.setClientMode()
NetworkTable.initialize()

file_name = os.path.join(args.output_dir, "structure.jpg")
tmp_name = os.path.join(args.output_dir, "structure.tmp.jpg")
vision = Vision()
vision.set_mode(8)
vision.setup_mode_listener()

logger = DataLogger(args.log_dir)

now = time.time()

with vision:
    while True:
        vision.get_depths()
        vision.process_depths()
        cv2.imwrite(tmp_name, vision.display)
        os.rename(tmp_name, file_name)
        logger.log_data(vision.depth, vision.ir)
        #time.sleep(0.05)
