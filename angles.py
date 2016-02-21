# for each pixel p[i] of structure data, what is the horizontal angle
# between the look vector and the pixel vector?
# assume fov is 58 degrees, equation is linear, and i ranges from 0 to N
# (N is either 320 or 640)
# assume look vector is same as pixel vector for i=CX 
# (probably pixel N/2, but don't assume that)
# then 
#  f(N) - f(0) = 58
#  f(CX) = 0, 0 <= CX <= N
#
# 58 = m*N + b - m*0 - b = m*N
# m = 58. / N
#
# 0 = m*CX + b
# b = -m*CX
#
# so
# angle = m*i + b


def h_angle(i, CX=None, N=320):
    if CX is None:
        CX = N / 2
    m = 58. / N
    b = -m * CX
    return m * i + b


# similar for vertical angle
def v_angle(i, CY=None, N=240):
    if CY is None:
        CY = N / 2
    m = 45. / N
    b = -m * CY
    return m * i + b


def test():
    assert h_angle(0) == -29
    assert h_angle(150) == -1.8125
    assert h_angle(160) == 0
    assert h_angle(170) == 1.8125
    assert h_angle(320) == 29

    assert v_angle(0) == -22.5
    assert v_angle(120) == 0
    assert v_angle(240) == 22.5


if __name__ == '__main__':
    test()
