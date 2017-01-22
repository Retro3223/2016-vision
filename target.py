import cv2

from utils import (
    mm_to_in
)


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
        return mm_to_in(self.avg_mm)

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
