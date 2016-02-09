import structure3223
import pygrip
import cv2
import numpy
from networktables import NetworkTable


GUI_NORMAL = 0x10


def main():
    # display detected goal and distance from it
    # in a live window
    cv2.namedWindow("View")
    cv2.createTrackbar("mode", "View", 0, 4, lambda *args: None)
    cv2.createTrackbar("area_threshold", "View", 10, 500,
                       lambda *args: None)
    with Vision() as vision:
        while True:
            vision.mode = cv2.getTrackbarPos("mode", "View")
            vision.area_threshold = cv2.getTrackbarPos("area_threshold", "View")
            vision.get_depths()
            vision.idepth_stats()
            vision.set_display()
            cv2.imshow("View", vision.display)
            x = cv2.waitKey(250)
            if x % 128 == 27:
                break
            elif 49 <= x <= 53:
                cv2.setTrackbarPos("mode", "View", x - 49)
        cv2.destroyWindow("View")


class Vision:
    def __init__(self):
        self.setup_nt()
        self.display = None
        self.mask8 = None
        self.mask16 = None
        self.depth = None
        self.interesting_depths = None
        self.ir = None
        self.contour_img = None
        self.mode = 0
        self.area_threshold = 10
        self.rat_min = 0.1
        self.rat_max = 0.6
        self.perimeter_threshold = 10
        self.n_shiniest = 1
        self.contours = []

    def __enter__(self):
        structure3223.init()
        return self

    def __exit__(self, *args):
        structure3223.destroy()

    def set_display(self):
        if self.mode == 0:
            display = to_uint8(self.depth)
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        elif self.mode == 1:
            display = cv2.cvtColor(to_uint8(self.ir), cv2.COLOR_GRAY2BGR)
        elif self.mode == 2:
            display = cv2.cvtColor(self.mask8, cv2.COLOR_GRAY2BGR)
        elif self.mode == 3:
            display = to_uint8(self.interesting_depths)
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        elif self.mode == 4:
            display = self.contour_img
        else:
            display = self.contour_img
        self.display = display

    def setup_nt(self):
        NetworkTable.setIPAddress('127.0.01')
        NetworkTable.setClientMode()
        NetworkTable.initialize()
        self.sd = NetworkTable.getTable("SmartDashboard")

    def get_depths(self):
        self.depth, self.ir = structure3223.read_frame()
        self.mask_shiny()
        self.filter_shiniest()
        self.interesting_depths = cv2.bitwise_and(self.depth, self.mask16)

    def mask_shiny(self):
        img2 = pygrip.desaturate(self.ir)
        img2_1 = to_uint8(img2)
        img3 = pygrip.blur(img2_1, pygrip.MEDIAN_BLUR, 1)
        _, img4 = cv2.threshold(img3, 80, 0xff, cv2.THRESH_BINARY)
        # grr threshold operates on matrices of unsigned bytes
        self.mask8 = img4
        self.mask16 = to_uint16_mask(img4)

    def filter_shiniest(self):
        things = cv2.findContours(
            self.mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = things[1]
        self.contour_img = cv2.cvtColor(self.mask8, cv2.COLOR_GRAY2BGR)
        # show all contours in blue
        cv2.drawContours(self.contour_img, contours, -1, (255, 0, 0))
        contours = [c for c in contours if self.filter(c)]
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
        count = len(depth_ixs[0])
        if count != 0:
            sum = numpy.sum(self.interesting_depths[depth_ixs])
            avg_mm = sum / count
            avg_in = avg_mm / 25.4
            avg_ft = avg_in / 12.
        else:
            avg_mm = -1
            avg_in = -1
            avg_ft = -1
        # self.sd.putNumber('d_mm', avg_mm)
        # self.sd.putNumber('d_in', avg_in)
        # self.sd.putNumber('d_ft', avg_ft)
        rects = [cv2.boundingRect(c) for c in self.contours]
        if len(rects) != 0:
            min_x = min([x for (x, y, w, h) in rects])
            max_x = max([x+w for (x, y, w, h) in rects])
            min_y = min([y for (x, y, w, h) in rects])
            max_y = max([y+h for (x, y, w, h) in rects])
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
            # crosshairs
            cv2.line(self.contour_img,
                     (center_x, center_y-5), (center_x, center_y+5), (0, 0, 255))
            cv2.line(self.contour_img,
                     (center_x-5, center_y), (center_x+5, center_y), (0, 0, 255))
            # corners
            cv2.line(self.contour_img,
                     (min_x, min_y), (min_x+5, min_y), (0, 0, 255))
            cv2.line(self.contour_img,
                     (min_x, min_y), (min_x, min_y+5), (0, 0, 255))
            cv2.line(self.contour_img,
                     (min_x, max_y), (min_x+5, max_y), (0, 0, 255))
            cv2.line(self.contour_img,
                     (min_x, max_y), (min_x, max_y-5), (0, 0, 255))
            cv2.line(self.contour_img,
                     (max_x, max_y), (max_x-5, max_y), (0, 0, 255))
            cv2.line(self.contour_img,
                     (max_x, max_y), (max_x, max_y-5), (0, 0, 255))
            cv2.line(self.contour_img,
                     (max_x, min_y), (max_x-5, min_y), (0, 0, 255))
            cv2.line(self.contour_img,
                     (max_x, min_y), (max_x, min_y+5), (0, 0, 255))
        else:
            min_y = -1
            max_y = -1
            min_x = -1
            max_x = -1
            center_x = -1
            center_y = -1
        cv2.putText(self.contour_img, ("d= %.2f in" % avg_in), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
        cv2.putText(self.contour_img, ("minx= %s" % min_x), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
        cv2.putText(self.contour_img, ("maxx= %s" % max_x), (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
        cv2.putText(self.contour_img, ("cx= %s" % center_x), (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))


def to_uint8(img):
    # convert a matrix of unsigned short values to a matrix of unsigned byte
    # values.
    # mostly just useful for turning sensor output into viewable images
    max = numpy.amax(img)
    result = img
    if max > 255:
        result = img * 255. / max
    result = result.astype('uint8')
    return result


def to_uint16_mask(img):
    # input should be uint8
    assert img.dtype == 'uint8'

    # we want to be able to bitwise_and the result of this function against
    # a matrix of unsigned shorts. without losing data.
    img5 = img.astype('uint16')
    img5[numpy.nonzero(img5)] = 0xffff
    return img5


if __name__ == '__main__':
    main()
