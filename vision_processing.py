import structure3223
import argparse
import math
import pygrip
import cv2
import numpy
from networktables import NetworkTable
from angles import (
    h_angle,
    v_angle
)
from data_logger import DataLogger, Replayer
from xyz_converter import depth_to_xyz


def setup_options_parser():
    parser = argparse.ArgumentParser(
        description='display structure sensor data.')
    parser.add_argument(
        '--replay-dir', dest='replay_dir', metavar='LDIR',
        default=None,
        help='specify directory of data to replay ' +
             '(or don\'t specify and display live sensor')
    parser.add_argument(
        '--record', dest='record', default=False, action='store_true',
        help='enable recording of data read from sensor')
    return parser

GUI_NORMAL = 0x10


def main():
    # display detected goal and distance from it
    # in a live window
    parser = setup_options_parser()
    args = parser.parse_args()
    replaying = args.replay_dir is not None
    recording = args.record
    if replaying:
        replayer = Replayer(args.replay_dir)
        mode = "stopped"
        top = 120
        left = 160
        cv2.namedWindow("View")
        cv2.createTrackbar("mode", "View", 0, 7, lambda *args: None)
        cv2.createTrackbar("area_threshold", "View", 10, 500,
                        lambda *args: None)
        cv2.createTrackbar("frame", "View", 0, len(replayer.frame_names), lambda *args: None)
        with Vision(use_sensor=False) as vision:
            while True:
                vision.mode = cv2.getTrackbarPos("mode", "View")
                vision.area_threshold = cv2.getTrackbarPos("area_threshold", "View")
                _frame_i = cv2.getTrackbarPos("frame", "View")
                if 0 <= _frame_i < len(replayer.frame_names):
                    frame_i = _frame_i
                vision.get_recorded_depths(replayer, frame_i)
                vision.idepth_stats()
                vision.set_display()
                if mode == "stopped" and vision.mode == 4:
                    cv2.rectangle(
                        vision.display, (left, top), (left + 10, top+10), 
                        (255, 0, 0))

                cv2.imshow("View", vision.display)
                wait_delay = 50
                if mode == "fw" and frame_i < len(replayer.frame_names) - 1:
                    cv2.setTrackbarPos("frame", "View", frame_i+1)
                    wait_delay = replayer.offset_milis(frame_i)
                elif mode == "bw" and 0 < frame_i:
                    cv2.setTrackbarPos("frame", "View", frame_i-1)
                    wait_delay = replayer.offset_milis(frame_i-1)
                x = cv2.waitKey(wait_delay)
                if x % 128 == 27:
                    break
                elif ord('0') <= x <= ord('7'):
                    cv2.setTrackbarPos("mode", "View", x - ord('0'))
                elif ord('`') == x:
                    cv2.setTrackbarPos("mode", "View", 0)
                elif ord('s') == x:
                    mode = "stopped"
                elif ord('f') == x:
                    mode = 'fw'
                elif ord('b') == x:
                    mode = 'bw'
                elif ord('p') == x:
                    print(replayer.file_name(frame_i))

                if mode == "stopped" and vision.mode == 4:
                    if x == 65361:
                        # left arrow key
                        left -= 2
                    elif x == 65362:
                        # up arrow key
                        top -= 2
                    elif x == 65363:
                        # right arrow key
                        left += 2
                    elif x == 65364:
                        # down arrow key
                        top += 2
                    elif x == ord('p'):
                        print('x: ', vision.xyz[0, top:top+10, left:left+10])
                        print('y: ', vision.xyz[1, top:top+10, left:left+10])
                        print('z: ', vision.xyz[2, top:top+10, left:left+10])
            cv2.destroyWindow("View")
    else:
        logger = DataLogger("logs")
        if recording:
            logger.begin_logging()
        cv2.namedWindow("View")
        cv2.createTrackbar("mode", "View", 0, 7, lambda *args: None)
        cv2.createTrackbar("area_threshold", "View", 10, 500,
                        lambda *args: None)
        with Vision() as vision:
            while True:
                vision.mode = cv2.getTrackbarPos("mode", "View")
                vision.area_threshold = cv2.getTrackbarPos("area_threshold", "View")
                vision.get_depths()
                vision.idepth_stats()
                vision.set_display()
                logger.log_data(vision.depth, vision.ir)
                cv2.imshow("View", vision.display)
                x = cv2.waitKey(50)
                if x % 128 == 27:
                    break
                elif ord('0') <= x <= ord('7'):
                    cv2.setTrackbarPos("mode", "View", x - ord('0'))
                elif ord('`') == x:
                    cv2.setTrackbarPos("mode", "View", 0)
            cv2.destroyWindow("View")


class Target:
    def __init__(self, min_x, max_x, min_y, max_y, contours):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.contours = contours
        self.draw_len = 5
        self.draw_thickness = 2
        self.draw_color = (0, 0, 255)
        self.avg_mm = -1
        self.theta = 1000
        self.theta_v = 1000

    @property
    def center_x(self):
        return (self.min_x + self.max_x) // 2

    @property
    def center_y(self):
        return (self.min_y + self.max_y) // 2

    @property
    def avg_in(self):
        return self.avg_mm / 25.4

    @property
    def avg_ft(self):
        return self.avg_in / 12.

    @staticmethod
    def merge(target1, target2):
        return Target(
            min_x=min(target1.min_x, target2.min_x),
            max_x=max(target1.max_x, target2.max_x),
            min_y=min(target1.min_y, target2.min_y),
            max_y=max(target1.max_y, target2.max_y),
            contours=target1.contours + target2.contours,
        )

    def line(self, pt1, pt2, img):
        cv2.line(img, pt1, pt2, self.draw_color, self.draw_thickness)

    def draw(self, img):
        LN = self.draw_len
        cx = self.center_x
        cy = self.center_y
        min_x, max_x = self.min_x, self.max_x
        min_y, max_y = self.min_y, self.max_y
        # crosshairs
        self.line((cx, cy-LN), (cx, cy+LN), img)
        self.line((cx-LN, cy), (cx+LN, cy), img)
        # corners
        self.line((min_x, min_y), (min_x+LN, min_y), img)
        self.line((min_x, min_y), (min_x, min_y+LN), img)
        self.line((min_x, max_y), (min_x+LN, max_y), img)
        self.line((min_x, max_y), (min_x, max_y-LN), img)
        self.line((max_x, max_y), (max_x-LN, max_y), img)
        self.line((max_x, max_y), (max_x, max_y-LN), img)
        self.line((max_x, min_y), (max_x-LN, min_y), img)
        self.line((max_x, min_y), (max_x, min_y+LN), img)

class Vision:
    def __init__(self, shape=(240, 320), use_sensor=True):
        shape3 = (shape[0], shape[1], 3)
        self.depth = numpy.zeros(shape=shape, dtype='uint16')
        self.ir = numpy.zeros(shape=shape, dtype='uint16')
        self.tmp16_1 = numpy.zeros(shape=shape, dtype='uint16')
        self.display = numpy.zeros(shape=shape3, dtype='uint8')
        self.tmp8_1 = numpy.zeros(shape=shape, dtype='uint8')
        self.tmp8_2 = numpy.zeros(shape=shape, dtype='uint8')
        self.mask8 = numpy.zeros(shape=shape, dtype='uint8')
        self.mask16 = numpy.zeros(shape=shape, dtype='uint16')
        self.interesting_depths = numpy.zeros(shape=shape, dtype='uint16')
        self.target_mask = numpy.zeros(shape=shape, dtype='uint16')
        self.target_depths = numpy.zeros(shape=shape, dtype='uint16')
        self.tmp83_1 = numpy.zeros(shape=shape3, dtype='uint8')
        self.contour_img = numpy.zeros(shape=shape3, dtype='uint8')
        self.xyz = numpy.zeros(shape=(3, shape[0], shape[1]), dtype='float32')
        self.mode = 0
        self.area_threshold = 10
        self.rat_min = 0.1
        self.rat_max = 0.6
        self.perimeter_threshold = 10
        self.n_shiniest = 1
        self.contours = []
        self.targets = []
        self.max_target_width = 70
        self.max_target_height = 70
        self.max_target_count = 3
        self.center_x = 160
        self.center_y = 120
        self.avg_mm = -1
        self.min_dist = 1000
        self.sd = NetworkTable.getTable("SmartDashboard")
        self.use_sensor = use_sensor

    def __enter__(self):
        if self.use_sensor:
            structure3223.init()
        return self

    def __exit__(self, *args):
        if self.use_sensor:
            structure3223.destroy()

    def set_display(self):
        if self.mode == 0:
            into_uint8(self.depth, dst=self.tmp8_1)
            cv2.cvtColor(self.tmp8_1, cv2.COLOR_GRAY2BGR, dst=self.display)
        elif self.mode == 1:
            into_uint8(self.ir, dst=self.tmp8_1)
            cv2.cvtColor(self.tmp8_1, cv2.COLOR_GRAY2BGR, dst=self.display)
        elif self.mode == 2:
            cv2.cvtColor(self.mask8, cv2.COLOR_GRAY2BGR, dst=self.display)
        elif self.mode == 3:
            into_uint8(self.interesting_depths, dst=self.tmp8_1)
            cv2.cvtColor(self.tmp8_1, cv2.COLOR_GRAY2BGR, dst=self.display)
        elif self.mode == 4:
            munge_floats_to_img(self.xyz, dst=self.display)
        else:
            numpy.copyto(dst=self.display, src=self.contour_img)

    def setup_mode_listener():
        self.sd.addTableListener(self.value_changed)

    def value_changed(self, table, key, value, is_new):
        if key == "structureMode":
            if value in [0, 1, 2, 3, 4, 5]:
                self.mode = value

    def get_depths(self):
        structure3223.read_frame(depth=self.depth, ir=self.ir)
        self.flip_inputs()
        self.zero_out_min_dists()
        self.mask_shiny()
        self.filter_shiniest()
        cv2.bitwise_and(self.depth, self.mask16, dst=self.interesting_depths)

    def get_recorded_depths(self, replayer, i):
        results = replayer.load_frame(i)
        self.depth, self.ir = results['depth'], results['ir']
        if 'xyz' in results:
            self.xyz = results['xyz']
        self.zero_out_min_dists()
        self.mask_shiny()
        self.filter_shiniest()
        cv2.bitwise_and(self.depth, self.mask16, dst=self.interesting_depths)

    def flip_inputs(self):
        cv2.flip(self.depth, 1, dst=self.depth)
        cv2.flip(self.ir, 1, dst=self.ir)

    def zero_out_min_dists(self):
        ixs = self.depth < self.min_dist
        self.depth[ixs] = 0
        self.ir[ixs] = 0

    def mask_shiny(self):
        pygrip.desaturate(self.ir, dst=self.tmp16_1)
        into_uint8(self.tmp16_1, dst=self.tmp8_1)
        pygrip.blur(self.tmp8_1, pygrip.MEDIAN_BLUR, 1, dst=self.tmp8_2)
        cv2.threshold(self.tmp8_2, 80, 0xff, cv2.THRESH_BINARY, dst=self.mask8)
        # grr threshold operates on matrices of unsigned bytes
        into_uint16_mask(self.mask8, dst=self.mask16)

    def filter_shiniest(self):
        things = cv2.findContours(
            self.mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = things[1]
        into_uint8(self.depth, dst=self.tmp8_1)
        cv2.cvtColor(self.tmp8_1, cv2.COLOR_GRAY2BGR, dst=self.contour_img)
        # show all contours in blue
        cv2.drawContours(self.contour_img, contours, -1, (255, 0, 0), cv2.FILLED)
        contours = [c for c in contours if self.filter(c)]
        contours.sort(key=lambda c: cv2.contourArea(c))
        cv2.drawContours(self.contour_img, contours, -1, (0, 255, 0))
        self.mask16[:] = 0
        self.mask8[:] = 0
        cv2.drawContours(self.mask8, contours, -1, (0xff), cv2.FILLED)
        cv2.drawContours(self.mask16, contours, -1, (0xffff), cv2.FILLED)
        self.contours = contours

    def filter(self, contour):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < self.area_threshold:
            return False
        ratio = float(perimeter) / float(area)
        if not (self.rat_min < ratio < self.rat_max):
            return False
        return True

    def idepth_stats(self):
        # compute some interesting things given a matrix
        # of "interesting" distances (noninteresting distances are 0)
        # idepth is a 240 x 320 matrix of depth data
        depth_ixs = numpy.nonzero(self.interesting_depths)
        depth_to_xyz(depth=self.depth, xyz=self.xyz)
        count = len(depth_ixs[0])
        if count != 0:
            sum = numpy.sum(self.interesting_depths[depth_ixs])
            self.avg_mm = sum / count
        else:
            avg_mm = -1
        rects = [(cv2.boundingRect(c), c) for c in self.contours]
        self.targets = self.build_targets(rects)

        chosen_j = self.choose_target()

        cv2.line(self.contour_img,
            (self.center_x-10, self.center_y),
            (self.center_x+10, self.center_y),
            (0, 0, 255), 2)
        cv2.line(self.contour_img,
            (self.center_x, self.center_y-10),
            (self.center_x, self.center_y+10),
            (0, 0, 255), 2)


        for j, target in enumerate(self.targets):
            if j == chosen_j:
                target.draw_color = (0, 255, 0)
            target.draw(self.contour_img)
            self.target_mask[:] = 0
            cv2.drawContours(
                self.target_mask,
                target.contours, -1, (0xffff), cv2.FILLED)
            cv2.bitwise_and(
                self.depth, self.target_mask, dst=self.target_depths)
            depth_ixs = numpy.nonzero(self.target_depths)
            count = len(depth_ixs[0])
            if count != 0:
                sum = numpy.sum(self.target_depths[depth_ixs])
                target.avg_mm = sum / count
                target.theta = h_angle(target.center_x, CX=self.center_x)
                target.theta_v = v_angle(target.center_y, CY=self.center_y)

            if j == chosen_j:
                self.sd.putNumber("target_dist", target.avg_mm)
                self.sd.putNumber("target_theta", target.theta)
                self.sd.putNumber("target_theta_v", target.theta_v)
                cv2.putText(self.contour_img,
                    ("d= %.2f in" % target.avg_in), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
                cv2.putText(self.contour_img,
                    ("theta= %.2f" % target.theta), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
                cv2.putText(self.contour_img,
                    ("thetav= %.2f" % target.theta_v), (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))

    @property
    def avg_ft(self):
        if self.avg_mm == -1:
            return -1
        return avg_mm / 25.4 / 12

    @property
    def avg_in(self):
        if self.avg_mm == -1:
            return -1
        return avg_mm / 25.4

    def build_targets(self, rects):
        targets = []
        target = None
        for rect, contour in rects:
            (x, y, w, h) = rect
            proposed_target = Target(x, x+w, y, y+h, [contour])
            i = -1
            for j, target in enumerate(targets):
                merged_target = Target.merge(target, proposed_target)
                if self.ok_target(merged_target):
                    targets[j] = merged_target
                    break
            else:
                if len(targets) < self.max_target_count:
                    targets.append(proposed_target)
        return targets

    def ok_target(self, target):
        return (target.max_x - target.min_x < self.max_target_width and
                target.max_y - target.min_y < self.max_target_height)

    def choose_target(self):
        target_dists = []
        for i, target in enumerate(self.targets):
            dist = math.hypot(
                (self.center_x - target.center_x),
                (self.center_y - target.center_y))
            target_dists.append((i, dist))

        target_dists.sort(key=lambda x: x[1])
        if len(target_dists) != 0:
            return target_dists[0][0]

    def crosshair(self, pt1, pt2, img=None):
        if img is None:
            img = self.contour_img
        corner_color = (0, 0, 255)
        corner_thickness = 2
        cv2.line(img, pt1, pt2, corner_color, corner_thickness)


def into_uint8(img, dst):
    # convert a matrix of unsigned short values to a matrix of unsigned byte
    # values.
    # mostly just useful for turning sensor output into viewable images
    max = numpy.amax(img)
    if max > 255:
        dst[:] = img * 255. / max
    else:
        numpy.copyto(dst, img)
    return dst


def into_uint16_mask(img, dst):
    # input should be uint8
    assert img.dtype == 'uint8'
    assert dst.dtype == 'uint16'

    # we want to be able to bitwise_and the result of this function against
    # a matrix of unsigned shorts. without losing data.
    dst[:] = 0
    dst[numpy.nonzero(img)] = 0xffff
    return dst


def munge_floats_to_img(xyz, dst):
    for i in range(0, 3):
        max = xyz[i, :,:].max()
        if max == 0:
            dst[:,:,i] = 0
        else:
            dst[:,:,i] = xyz[i,:,:] * 255. / xyz[i, :,:].max()
    return dst


if __name__ == '__main__':
    main()
