import numpy
import math

horizontal_fov = 1.02586
vertical_fov = 0.799344 

nx_cache = {}
ny_cache = {}


def depth_to_xyz(depth, xyz, shape=(240, 320)):
    if shape not in nx_cache:
        ix = numpy.indices(shape)
        xz_factor = math.tan(horizontal_fov / 2) * 2
        yz_factor = math.tan(vertical_fov / 2) * 2
        nx = (ix[1] / float(shape[1]) - 0.5) * xz_factor
        ny = (0.5 - ix[0] / float(shape[0])) * yz_factor
        nx_cache[shape] = nx
        ny_cache[shape] = ny
    else:
        nx = nx_cache[shape]
        ny = ny_cache[shape]
    xyz[0, :] = nx * depth
    xyz[1, :] = ny * depth
    xyz[2, :] = depth

    return xyz


def depth_to_xyz2(depth, xyz, shape=(240, 320)):
    j, i = numpy.indices(shape)
    vangle = vertical_fov * (-j/240. + 0.5)
    hangle = horizontal_fov * (i/320. - 0.5)
    xyz[0] = depth * numpy.cos(vangle) * numpy.sin(hangle)
    xyz[1] = depth * numpy.sin(vangle) 
    xyz[2] = depth * numpy.cos(vangle) * numpy.cos(hangle)
    return xyz


def distance(pt1, pt2):
    """
    distance between points pt1 and pt2, which are points in 2d or 3d 
    cartesian coordinate space
    """
    from scipy.spatial.distance import pdist
    dim = len(pt1)
    pts = numpy.empty((2, dim), dtype='float32')
    pts[0] = pt1
    pts[1] = pt2
    return pdist(pts)[0]


def midpoint(pt1, pt2):
    return ((pt1[0]+pt2[0])*0.5, (pt1[1]+pt2[1])*0.5)
