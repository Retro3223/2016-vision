import argparse
import time
import cv2
from data_logger import DataLogger, Replayer
from vision_processing import (
    Vision
)


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

def scale_image(img, factor):
    if factor == 1:
        return img
    newsize = (img.shape[1]*factor, img.shape[0]*factor)
    return cv2.resize(img, newsize)

def main():
    # display detected goal and distance from it
    # in a live window
    max_mode = 8
    parser = setup_options_parser()
    args = parser.parse_args()
    replaying = args.replay_dir is not None
    recording = args.record
    pclviewer = None
    if replaying:
        replayer = Replayer(args.replay_dir)
        mode = "stopped"
        cv2.namedWindow("View")
        cv2.createTrackbar("mode", "View", 0, max_mode, lambda *args: None)
        cv2.createTrackbar("image_scale", "View", 1, 10,
                        lambda *args: None)
        cv2.createTrackbar("position", "View", 0, 1,
                        lambda *args: None)
        cv2.createTrackbar("frame", "View", 0, len(replayer.frame_names), lambda *args: None)
        cv2.setTrackbarPos("mode", "View", 8)
        cv2.setTrackbarPos("position", "View", 1)
        cv2.setTrackbarPos("image_scale", "View", 3)
        scale_factor = 3
        with Vision(use_sensor=False) as vision:
            def on_mouse(ev, x, y, flags, userdata):
                return vision.on_mouse(ev, x // scale_factor, y // scale_factor, flags, userdata)

            cv2.setMouseCallback("View", on_mouse, None)
            while True:
                vision.set_mode(cv2.getTrackbarPos("mode", "View"))
                _frame_i = cv2.getTrackbarPos("frame", "View")
                if 0 <= _frame_i < len(replayer.frame_names):
                    frame_i = _frame_i
                vision.get_recorded_depths(replayer, frame_i)
                vision.process_depths()
                if mode != "stopped" and pclviewer is not None:
                    pclviewer.updateCloud(vision.xyz)
                position = cv2.getTrackbarPos("position", "View")
                if position == 0:
                    vision.is_hg_position = True
                    vision.is_gear_position = False
                else:
                    vision.is_hg_position = False
                    vision.is_gear_position = True


                scale_factor = cv2.getTrackbarPos("image_scale", "View")
                if scale_factor == 0:
                    scale_factor = 1
                img_to_show = scale_image(vision.display, scale_factor)
                cv2.imshow("View", img_to_show)
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
                elif ord('0') <= x <= ord(str(max_mode)):
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
                elif ord('i') == x:
                    cv2.imwrite("plop.jpg", vision.display);
                elif ord('z') == x and libpclproc is not None:
                    if pclviewer is None:
                        pclviewer = libpclproc.process(vision.xyz)
                    else:
                        pclviewer.close()
                        pclviewer = None
                if pclviewer is not None and not pclviewer.wasStopped():
                    pclviewer.spin()

            cv2.destroyWindow("View")
    else:
        logger = DataLogger("logs")
        lasttime = None
        if recording:
            logger.begin_logging()
        cv2.namedWindow("View")
        cv2.createTrackbar("mode", "View", 0, max_mode, lambda *args: None)
        cv2.createTrackbar("angle", "View", 0, 90,
                        lambda *args: None)
        cv2.createTrackbar("position", "View", 0, 1,
                        lambda *args: None)
        with Vision() as vision:
            cv2.setMouseCallback("View", vision.on_mouse, None)
            while True:
                vision.set_mode(cv2.getTrackbarPos("mode", "View"))
                vision.angle = cv2.getTrackbarPos("angle", "View")
                position = cv2.getTrackbarPos("position", "View")
                if position == 0:
                    vision.is_hg_position = True
                    vision.is_gear_position = False
                else:
                    vision.is_hg_position = False
                    vision.is_gear_position = True
                vision.get_depths()
                t1 = time.time()
                if lasttime is not None: print(t1-lasttime)
                lasttime= t1
                vision.process_depths()
                t2 = time.time()
                logger.log_data(vision.depth, vision.ir)
                cv2.imshow("View", vision.display)
                x = cv2.waitKey(50)
                if x % 128 == 27:
                    break
                elif ord('0') <= x <= ord(str(max_mode)):
                    cv2.setTrackbarPos("mode", "View", x - ord('0'))
                elif ord('`') == x:
                    cv2.setTrackbarPos("mode", "View", 0)
            cv2.destroyWindow("View")


if __name__ == '__main__':
    main()
