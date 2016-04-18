from unittest import TestCase
import numpy
from xyz_converter import depth_to_xyz
from xyz_converter import depth_to_xyz2

class XYZTests(TestCase):
    def test_convert_to_xyz(self):
        depths = numpy.empty(shape=(240, 320), dtype='uint16')
        xyz = numpy.zeros(shape=(3, 240, 320), dtype='float32')
        depths[:, :] = 1766
        xyz = depth_to_xyz(depths, xyz)
        self.assertAlmostEqual(xyz[0, 120, 160], 0)
        self.assertAlmostEqual(xyz[1, 120, 160], 0)
        self.assertAlmostEqual(xyz[2, 120, 160], 1766)
        
        self.assertAlmostEqual(xyz[0, 0, 0], -994.63201904296875)
        self.assertAlmostEqual(xyz[1, 0, 0], 745.97015380859375)
        self.assertAlmostEqual(xyz[2, 0, 0], 1766) # really?

    def test_convert_to_xyz2(self):
        depths = numpy.empty(shape=(240, 320), dtype='uint16')
        xyz = numpy.zeros(shape=(3, 240, 320), dtype='float32')
        depths[:, :] = 1766
        xyz = depth_to_xyz2(depths, xyz)
        self.assertAlmostEqual(xyz[0, 120, 160], 0)
        self.assertAlmostEqual(xyz[1, 120, 160], 0)
        self.assertAlmostEqual(xyz[2, 120, 160], 1766)

        self.assertAlmostEqual(xyz[0, 120, 0], -866.63322, 4)
        self.assertAlmostEqual(xyz[1, 120, 0], 0)
        self.assertAlmostEqual(xyz[2, 120, 0], 1538.7341, 4)

        self.assertAlmostEqual(xyz[0, 120, 319], 861.69586, 4)
        self.assertAlmostEqual(xyz[1, 120, 319], 0)
        self.assertAlmostEqual(xyz[2, 120, 319], 1541.5045, 4)

        self.assertAlmostEqual(xyz[0, 0, 160], 0)
        self.assertAlmostEqual(xyz[1, 0, 160], 687.1793, 4)
        self.assertAlmostEqual(xyz[2, 0, 160], 1626.8192, 4)

        self.assertAlmostEqual(xyz[0, 239, 160], 0)
        self.assertAlmostEqual(xyz[1, 239, 160], -681.7571, 4)
        self.assertAlmostEqual(xyz[2, 239, 160], 1629.0989, 4)
        
        self.assertAlmostEqual(xyz[0, 0, 0], -798.3327, 4)
        self.assertAlmostEqual(xyz[1, 0, 0], 687.17926, 4)
        self.assertAlmostEqual(xyz[2, 0, 0], 1417.4645, 4)

