from unittest import TestCase
import numpy
import math
from vision_processing import BestFitLine

class BestFitLineTests(TestCase):
    def test_line1(self):
        # goofy?
        ts = numpy.array([
            0.,
            12.04159458,
            14.56021978,
            6.32455532
        ])
        xs = numpy.array([
            167.,
            166.,
            163.,
            165.,
        ])
        ys = numpy.array([
            123.,
            111.,
            137.,
            129.,
        ])
        edges = list(zip(xs, ys))

        line = BestFitLine(edges)

        self.assertAlmostEqual(line.y_from_t(0), -276.818, 3)
        self.assertAlmostEqual(line.t_from_y(-276.818), 0.0, 3)

        self.assertAlmostEqual(line.x_from_t(0), 206.4785, 3)
        self.assertAlmostEqual(line.t_from_x(206.4785), 0.0, 3)

        self.assertAlmostEqual(line.y_from_t(12.04159458), -253.484852, 3)
        self.assertAlmostEqual(line.t_from_y(-253.484852), 12.04159458, 2)

        self.assertAlmostEqual(line.x_from_t(12.04159458), 204.0844, 3)
        self.assertAlmostEqual(line.t_from_x(204.0844), 12.04159458, 2)

    def test_slope1(self):
        pts = [(1, -1), (2, 2)]
        line = BestFitLine(pts)

        self.assertAlmostEqual(line.mx, math.sqrt(0.5), 3)
        self.assertAlmostEqual(line.my, math.sqrt(4.5), 3)

        slope = line.xy_slope()
        self.assertAlmostEqual(slope, 3, 3)

    def test_slope1_5(self):
        pts = [(1, -1), (0, 2)]
        line = BestFitLine(pts)

        self.assertAlmostEqual(line.mx, -1.707, 3)
        self.assertAlmostEqual(line.my, 5.121, 3)

        slope = line.xy_slope()
        self.assertAlmostEqual(slope, -3, 3)

    def test_slope2(self):
        pts = [(1, -1), (1, 2)]
        line = BestFitLine(pts)

        self.assertAlmostEqual(line.mx, 0, 3)
        self.assertAlmostEqual(line.my, 3.65, 3)

        slope = line.xy_slope()
        assert math.isinf(slope)

    def test_slope3(self):
        pts = [(1, 2), (2, 2)]
        line = BestFitLine(pts)

        self.assertAlmostEqual(line.mx, 1.688, 3)
        self.assertAlmostEqual(line.my, 0, 3)

        slope = line.xy_slope()
        self.assertAlmostEqual(slope, 0.0, 3)
