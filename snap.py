import structure3223
import pygrip
import time
import cv2
import numpy


def main():
    # display detected goal and distance from it
    # in a live window
    cv2.namedWindow("Example")
    while True:
        depth, ir, interesting_depth = get_depths()
        idepth_stats(interesting_depth)
        y = to_uint8(interesting_depth)
        idepthimg = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Example", idepthimg)
        if cv2.waitKey(15) % 128 == 27:
            break
        time.sleep(1)
    cv2.destroyWindow("Example")


def idepth_stats(idepth):
    # compute some interesting things given a matrix
    # of "interesting" distances (noninteresting distances are 0)
    # idepth is a 240 x 320 matrix of depth data
    depth_ixs = numpy.nonzero(idepth)
    count = len(depth_ixs[0])
    if count != 0:
        sum = numpy.sum(idepth[depth_ixs])
        avg_mm = sum / count
        avg_in = avg_mm / 25.4
        avg_ft = avg_in / 12.
        print("avg: %f mm" % avg_mm)
        print("avg: %f inches" % avg_in)
        print("avg: %f feet" % avg_ft)


def to_uint8(img):
    # convert a matrix of unsigned short values to a matrix of unsigned byte
    # values.
    # mostly just useful for turning sensor output into viewable images
    max = numpy.amax(img)
    result = img
    if max > 255:
        result = img * 255. / max
    result = result.astype('uint8')
    return result


def get_depths():
    depth, ir = structure3223.read_frame()
    mask = ir_mask(ir)
    interesting_depths = cv2.bitwise_and(depth, mask)
    # print (interesting_depths[numpy.where(interesting_depths != 0)])
    return depth, ir, interesting_depths


def ir_mask(ir_img):
    img2 = pygrip.desaturate(ir_img)
    img2_1 = to_uint8(img2)
    img3 = pygrip.blur(img2_1, pygrip.MEDIAN_BLUR, 3)
    _, img4 = cv2.threshold(img3, 100, 0xff, cv2.THRESH_BINARY)
    # grr threshold operates on matrices of unsigned bytes
    img5 = img4.astype('uint16')
    # we want to be able to bitwise_and the result of this function against
    # a matrix of unsigned shorts. without losing data.
    img5[numpy.nonzero(img5)] = 0xffff
    return img5


if __name__ == '__main__':
    structure3223.init()
    main()
    structure3223.destroy()
