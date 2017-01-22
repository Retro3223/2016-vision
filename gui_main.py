import argparse
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

def main():
    # display detected goal and distance from it
    # in a live window
    parser = setup_options_parser()
    args = parser.parse_args()
    replaying = args.replay_dir is not None
    recording = args.record
    pclviewer = None
    if replaying:
        replayer = Replayer(args.replay_dir)
        mode = "stopped"
        cv2.namedWindow("View")
        cv2.createTrackbar("mode", "View", 0, 7, lambda *args: None)
        cv2.createTrackbar("area_threshold", "View", 10, 500,
                        lambda *args: None)
        cv2.createTrackbar("frame", "View", 0, len(replayer.frame_names), lambda *args: None)
        with Vision(use_sensor=False) as vision:
            cv2.setMouseCallback("View", vision.on_mouse, None)
            while True:
                vision.set_mode(cv2.getTrackbarPos("mode", "View"))
                vision.area_threshold = cv2.getTrackbarPos("area_threshold", "View")
                _frame_i = cv2.getTrackbarPos("frame", "View")
                if 0 <= _frame_i < len(replayer.frame_names):
                    frame_i = _frame_i
                vision.get_recorded_depths(replayer, frame_i)
                vision.process_depths()
                if mode != "stopped" and pclviewer is not None:
                    pclviewer.updateCloud(vision.xyz)


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
        if recording:
            logger.begin_logging()
        cv2.namedWindow("View")
        cv2.createTrackbar("mode", "View", 0, 7, lambda *args: None)
        '''
        cv2.createTrackbar("area_threshold", "View", 10, 500,
                        lambda *args: None)
        '''
        cv2.createTrackbar("angle", "View", 0, 90,
                        lambda *args: None)
        cv2.createTrackbar("velocity", "View", 1000, 10000,
                        lambda *args: None)
        with Vision() as vision:
            cv2.setMouseCallback("View", vision.on_mouse, None)
            while True:
                vision.set_mode(cv2.getTrackbarPos("mode", "View"))
                #vision.area_threshold = cv2.getTrackbarPos("area_threshold", "View")
                vision.angle = cv2.getTrackbarPos("angle", "View")
                vision.get_depths()
                vision.process_depths()
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


if __name__ == '__main__':
    main()
