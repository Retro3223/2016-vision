from unittest import TestCase
import numpy
from utils import least_squares

class LeastSquaresTests(TestCase):
    def test_least_squares1(self):
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
        (b, m) = least_squares(ts, xs)

        xmean = xs.mean()
        tmean = ts.mean()
        self.assertAlmostEqual(xmean, 165.25, 3)
        self.assertAlmostEqual(tmean, 8.232, 3)

        num = (
            (ts[0] - tmean) * (xs[0] - xmean) + 
            (ts[1] - tmean) * (xs[1] - xmean) + 
            (ts[2] - tmean) * (xs[2] - xmean) + 
            (ts[3] - tmean) * (xs[3] - xmean)
        )
        den = (
            (ts[0] - tmean) ** 2 +
            (ts[1] - tmean) ** 2 +
            (ts[2] - tmean) ** 2 +
            (ts[3] - tmean) ** 2 
        )

        self.assertAlmostEqual(m, num / den, 3)


        self.assertAlmostEqual(b, 166.904, 3)
        self.assertAlmostEqual(m, -0.201, 3)

        (b, m) = least_squares(ts, ys)
        self.assertAlmostEqual(b, 122.9455, 3)
        self.assertAlmostEqual(m, 0.2495, 3)

    def test_least_squares2(self):
        ts = numpy.array([
            0., 
            13.34166406,
            6.08276253,
        ])
        xs = numpy.array([
            126.,
            123., 
            125.,
        ])
        ys = numpy.array([
            113.,
            126.,
            119.,
        ])
        (b, m) = least_squares(ts, xs)
        self.assertAlmostEqual(b, 126.133, 3)
        self.assertAlmostEqual(m, -0.226, 3)

        (b, m) = least_squares(ts, ys)
        self.assertAlmostEqual(b, 113.026, 3)
        self.assertAlmostEqual(m, 0.97407, 3)
