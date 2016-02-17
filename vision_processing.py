import structure3223
import pygrip
import cv2
import numpy


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
            x = cv2.waitKey(50)
            if x % 128 == 27:
                break
            elif 49 <= x <= 53:
                cv2.setTrackbarPos("mode", "View", x - 49)
        cv2.destroyWindow("View")


class Vision:
    def __init__(self, shape=(240, 320)):
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
        self.tmp83_1 = numpy.zeros(shape=shape3, dtype='uint8')
        self.contour_img = numpy.zeros(shape=shape3, dtype='uint8')
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
        else:
            numpy.copyto(dst=self.display, src=self.contour_img)

    def get_depths(self):
        structure3223.read_frame(depth=self.depth, ir=self.ir)
        self.flip_inputs()
        self.mask_shiny()
        self.filter_shiniest()
        cv2.bitwise_and(self.depth, self.mask16, dst=self.interesting_depths)

    def flip_inputs(self):
        cv2.flip(self.depth, 1, dst=self.depth)
        cv2.flip(self.ir, 1, dst=self.ir)

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
        rects = [cv2.boundingRect(c) for c in self.contours]
        if len(rects) != 0:
            min_x = min([x for (x, y, w, h) in rects])
            max_x = max([x+w for (x, y, w, h) in rects])
            min_y = min([y for (x, y, w, h) in rects])
            max_y = max([y+h for (x, y, w, h) in rects])
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
            # crosshairs
            self.crosshair(
                     (center_x, center_y-5),
                     (center_x, center_y+5))
            self.crosshair(
                     (center_x-5, center_y),
                     (center_x+5, center_y))
            # corners
            self.crosshair((min_x, min_y), (min_x+5, min_y))
            self.crosshair((min_x, min_y), (min_x, min_y+5))
            self.crosshair((min_x, max_y), (min_x+5, max_y))
            self.crosshair((min_x, max_y), (min_x, max_y-5))
            self.crosshair((max_x, max_y), (max_x-5, max_y))
            self.crosshair((max_x, max_y), (max_x, max_y-5))
            self.crosshair((max_x, min_y), (max_x-5, min_y))
            self.crosshair((max_x, min_y), (max_x, min_y+5))
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


if __name__ == '__main__':
    main()
