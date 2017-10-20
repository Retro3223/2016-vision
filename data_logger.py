#!/usr/bin/python

import os
import os.path
import time
import datetime
import numpy
from networktables import NetworkTable


class DataLogger:
    def __init__(self, log_dir):
        self.match_running = False
        self.time_start = None
        self.time_stop = None
        self.log_dir = log_dir
        self.save_dir = None
        self.log_period = 100 # ms
        self.last_log_time_milis = 0 # time, ms
        self.sd = NetworkTable.getTable("SmartDashboard")

        self.sd.addTableListener(self.value_changed)

    def value_changed(self, table, key, value, is_new):
        if key == "auto_time_remaining" and value == 15 and not self.match_running:
            self.begin_logging()
            self.stop_at(150)

    def begin_logging(self):
        self.time_stop = None
        self.match_running = True
        self.save_dir = os.path.join(
            self.log_dir,
            datetime.datetime.now().isoformat())
        try:
            os.makedirs(self.save_dir)
        except:
            # forget that idea, i guess
            self.match_running = False
        self.time_start = time.time()

    def stop_at(self, n):
        self.time_stop = self.time_start + n

    def stop(self):
        self.match_running = False

    def stop_when_done(self):
        now = time.time()
        if self.match_running and \
                self.time_stop is not None and \
                now > self.time_stop:
            self.stop()

    def log_data(self, depth, ir):
        if self.match_running:
            try:
                now_milis = int(time.time() * 1000)
                if now_milis - self.last_log_time_milis > self.log_period:
                    depthnom = os.path.join(self.save_dir, "%s_depth" % (now_milis,))
                    irnom = os.path.join(self.save_dir, "%s_ir" % (now_milis,))
                    numpy.save(file=depthnom, arr=depth)
                    numpy.save(file=irnom, arr=ir)
                    self.last_log_time_milis = now_milis
            except:
                # don't stop the main loop!
                raise
        self.stop_when_done()


def get_timestamp(filenom, suffix):
    if filenom.endswith(suffix):
        return filenom[:-len(suffix)]


class Replayer:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        files = [os.path.splitext(x)[0] for x in os.listdir(log_dir)]

        potential_frames = set()
        self.frame_names = []
        for item in files:
            nom = get_timestamp(item, '_depth')
            if nom:
                potential_frames.add(nom)
            else:
                nom = get_timestamp(item, '_ir')
                if nom and nom in potential_frames:
                    potential_frames.remove(nom)
                    self.frame_names.append(nom)
        self.frame_names.sort()

    def load_frame(self, i):
        _i = i
        ir, depth = None, None
        results = {}
        if isinstance(i, int):
            i = self.frame_names[i]
        try:
            depthnom = os.path.join(self.log_dir, "%s_depth.npy" % (i,))
            irnom = os.path.join(self.log_dir, "%s_ir.npy" % (i,))
            results['ir'] = numpy.load(irnom)
            results['depth'] = numpy.load(depthnom)
        except:
            print ('bad! ', _i, i)
            raise

        return results

    def offset_milis(self, i):
        if 0 <= i < len(self.frame_names)-1:
            milis_now = int(self.frame_names[i])
            milis_next = int(self.frame_names[i+1])
            return milis_next - milis_now

    def file_name(self, i):
        if 0 <= i < len(self.frame_names):
            return os.path.join(self.log_dir, "%s_depth.npy" % (self.frame_names[i],))


class LegacyReplayer:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.frame_names = [os.path.splitext(x)[0] for x in os.listdir(log_dir)]
        self.frame_names.sort()

    def load_frame(self, i):
        _i = i
        ir, depth = None, None
        results = {}
        if isinstance(i, int):
            i = self.frame_names[i]
        try:
            nom = os.path.join(self.log_dir, "%s.npz" % (i,))
            with numpy.load(nom) as recorded:
                for k in recorded.keys():
                    results[k] = recorded[k]
        except:
            print ('bad! ', _i, i)
            raise

        return results

    def offset_milis(self, i):
        if 0 <= i < len(self.frame_names)-1:
            milis_now = int(self.frame_names[i])
            milis_next = int(self.frame_names[i+1])
            return milis_next - milis_now

    def file_name(self, i):
        if 0 <= i < len(self.frame_names):
            return os.path.join(self.log_dir, self.frame_names[i]+".npz")


if __name__ == '__main__':
    replayer = Replayer("logs/pt2")
    replayer.load_frame(0)
