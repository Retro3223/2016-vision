import numpy
from utils import least_squares
from xyz_converter import distance


class BestFitLine:
    """
    given a set of (x, y) points, find a line of best fit.
    also, provide some convenience functions for finding points on the line
    """
    def __init__(self, points):
        pixel_xy0 = (0.0, 0.0)
        ts = numpy.zeros(shape=(len(points),), dtype='float')
        xs = numpy.zeros(shape=(len(points),), dtype='float')
        ys = numpy.zeros(shape=(len(points),), dtype='float')
        for i in range(len(points)): 
            pixel_xyi = points[i]
            ts[i] = distance(pixel_xyi, pixel_xy0)
            xs[i] = pixel_xyi[0]
            ys[i] = pixel_xyi[1]
        (self.bx, self.mx) = least_squares(ts, xs)
        (self.by, self.my) = least_squares(ts, ys)

    def y_from_t(self, t):
        return self.my * t + self.by

    def x_from_t(self, t):
        return self.mx * t + self.bx

    def t_from_y(self, y):
        return (y - self.by) / self.my

    def t_from_x(self, x):
        return (x - self.bx) / self.mx

    def xy_slope(self):
        x0 = self.x_from_t(0)
        y0 = self.y_from_t(0)
        x1 = self.x_from_t(1)
        y1 = self.y_from_t(1)

        if x0 == x1: return float('inf')
        return (y1 - y0) / (x1 - x0)
