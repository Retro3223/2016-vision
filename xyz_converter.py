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
    xyz[2, :] = depth;

    return xyz

def depth_to_xyz2(depth, xyz, shape=(240, 320)):
    vangle = VFOV * (-j/240 + 0.5)
    hangle = HFOV * (i/320 - 0.5)
    x = depth * cos(vangle) * sin(hangle)
    y = depth * sin(vangle) 
    z = depth * cos(vangle) * cos(hangle)
    pass
