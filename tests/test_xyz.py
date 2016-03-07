from unittest import TestCase
import numpy
from xyz_converter import depth_to_xyz

class XYZTests(TestCase):
    def test_convert_to_xyz(self):
        depths = numpy.empty(shape=(240, 320), dtype='uint16')
        xyz = numpy.zeros(shape=(3, 240, 320), dtype='float32')
        depths[:, :] = 1766
        xyz = depth_to_xyz(depths, xyz)
        self.assertAlmostEqual(xyz[0, 120, 160], 0)
        self.assertAlmostEqual(xyz[1, 120, 160], 0)
        self.assertAlmostEqual(xyz[2, 120, 160], 1766)
        
        print("%1.22f" % xyz[0,0,0])
        self.assertAlmostEqual(xyz[0, 0, 0], -994.63201904296875)
        self.assertAlmostEqual(xyz[1, 0, 0], 745.97015380859375)
        self.assertAlmostEqual(xyz[2, 0, 0], 1766) # really?

