from math import sin, cos, atan2
from typing import List
from apple import Apple
from xypoint import XYPoint
from mathutils import distance, line_distance


class Eye():
    def __init__(self, xy: XYPoint, baseAngle: float, eyeAngle: float, eyeWidth: float, length: float, screenWidth: int,
                 screenHeight: int):
        self.eyeAngle = eyeAngle
        self.angle = eyeAngle + baseAngle
        self.eyeWidth = eyeWidth
        self.length = length
        self.xy = xy
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        self.update()

    def update(self):
        self.p1 = self.xy
        self.p2 = self.project(self.angle, self.length)
        self.actual_length = distance(self.p1, self.p2)

    def move(self, xy: XYPoint, baseAngle: float):
        self.xy = xy
        self.angle = baseAngle + self.eyeAngle
        self.update()

    def project(self, angle: float, length: float) -> XYPoint:
        dx = cos(angle) * length
        dy = sin(angle) * length

        # have to clip end of eye sensor to the bounding box of the window

        if self.xy.x + dx > self.screenWidth:
            new_dx = self.screenWidth - self.p1.x
            dy = new_dx / dx * dy
            dx = new_dx
        if self.xy.x + dx < 0:
            new_dx = -self.p1.x
            dy = new_dx / dx * dy
            dx = new_dx
        if self.xy.y + dy > self.screenHeight:
            new_dy = self.screenHeight - self.p1.y
            dx = new_dy / dy * dx
            dy = new_dy
        if self.xy.y + dy < 0:
            new_dy = -self.xy.y
            dx = new_dy / dy * dx
            dy = new_dy

        return XYPoint(self.xy.x + dx, self.xy.y + dy)

    def computeAngle(self, p: XYPoint) -> float:
        pp = p - self.xy
        return atan2(pp.x, pp.y)
