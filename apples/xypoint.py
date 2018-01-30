import math

# 'S'mart Point, I can't stand that you can't add two 'Points' together.
class XYPoint(object):
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, p2: 'XYPoint') -> 'XYPoint':
        return XYPoint(self.x + p2.x, self.y + p2.y)

    def __sub__(self, p2: 'XYPoint') -> 'XYPoint':
        return XYPoint(self.x - p2.x, self.y - p2.y)

    # the angle from the origin: (0, 0) to the point (x, y)
    def angle(self):
        return math.atan2(self.y, self.x)
