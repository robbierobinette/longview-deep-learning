from math import sqrt, sin, cos
from xypoint import XYPoint
from math import pi


def distance(p1, p2) -> float:
    xd = p1.x - p2.x
    yd = p1.y - p2.y
    return sqrt(xd * xd + yd * yd)


def rotate(angle: float, p: XYPoint) -> XYPoint:
    cc = cos(angle)
    ss = sin(angle)

    xx = cc * p.x - ss * p.y
    yy = ss * p.x + cc * p.y
    return XYPoint(xx, yy)


def line_distance(l1: XYPoint, l2: XYPoint, p: XYPoint) -> float:
    dx = l2.x - l1.x
    dy = l2.y - l1.y
    if dx == 0.0 and dy == 0.0:
        return 100000
    else:
        return abs(dy * p.x - dx * p.y + l2.x * l1.y - l2.y * l1.x) / sqrt(dx * dx + dy * dy)


# reduces an angle measurement in radians to something in the range -pi -> pi
def reduce_angle(angle: float) -> float:
    if (angle < 0):
        n = int(-angle / (2 * pi)) + 1
        aa = angle + n * 2 * pi
    else:
        n = int(angle / (2 * pi))
        aa = angle - n * 2 * pi

    if (aa > pi):
        return aa - 2 * pi
    else:
        return aa
