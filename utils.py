import numpy
import cv2

def mm_to_in(mms):
    """
    convert millimeters to inches
    """
    return mms / 25.4


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


def munge_floats_to_img(xyz, dst):
    for i in range(0, 3):
        max = xyz[i, :,:].max()
        if max == 0:
            dst[:,:,i] = 0
        else:
            dst[:,:,i] = xyz[i,:,:] * 255. / xyz[i, :,:].max()
    return dst


def rgbhex2bgr(hexcolor):
    b = hexcolor & 0xff
    g = (hexcolor >> 8) & 0xff
    r = (hexcolor >> 16) & 0xff
    return (b, g, r)


def minAreaBox(contours):
    rect2 = cv2.minAreaRect(contours)
    box = cv2.boxPoints(rect2)
    box = numpy.int0(box)
    return [(x, y) for [x,y] in box]


def boxCenter(box):
    mid_x = int(sum([b[0] for b in box])/len(box))
    mid_y = int(sum([b[1] for b in box])/len(box))
    return (mid_x, mid_y)


def flatten_contours(contours):
    dim1 = sum([x.shape[0] for x in contours])
    flattened_contours = numpy.empty(shape=(dim1, 1, 2), dtype='int32')
    i = 0
    for contour in contours:
        cnt = contour.shape[0]
        flattened_contours[i:i+cnt, :, :] = contour
        i += cnt
    return flattened_contours

def least_squares(ts, xs):
    tmean = ts.mean()
    xmean = xs.mean()
    tg = ((ts - tmean)**2).sum()
    if tg == 0:
        print ('denominator 0?! ', ts)
    b = ((ts - tmean) * (xs - xmean)).sum() / ((ts - tmean)**2).sum()
    a = xmean - b * tmean
    return (a, b)
