import cv2
# import numpy as np


def channels(img):
    if len(img.shape) == 2:
        return 1
    return img.shape[2]


def desaturate(img):
    _channels = channels(img)
    if _channels == 1:
        return img
    elif _channels == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif _channels == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)


BOX_BLUR = 1
GAUSSIAN_BLUR = 2
MEDIAN_BLUR = 3
BILATERAL_FILTER = 4


def blur(img, type, radius):
    if type == BOX_BLUR:
        kdim = 2 * int(radius) + 1
        return cv2.blur(img, (kdim, kdim))
    elif type == GAUSSIAN_BLUR:
        kdim = 6 * int(radius) + 1
        return cv2.GaussianBlur(img, (kdim, kdim), radius)
    elif type == MEDIAN_BLUR:
        ksize = 2 * int(radius) + 1
        return cv2.medianBlur(img, ksize)
    elif type == BILATERAL_FILTER:
        return cv2.bilateralFilter(img, -1, radius, radius)
