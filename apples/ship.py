from math import *
from eye import Eye
from xypoint import XYPoint
from mathutils import distance, reduce_angle


class Ship():
    def __init__(self, xy: XYPoint, ship_angle: float, width: int, height: int, n_sensors: int, sensor_resolution: int, sensor_length: int, sensor_width: float):
        self.xy = xy
        self.angle = ship_angle

        self.width = width
        self.height = height
        self.n_sensors = n_sensors
        self.sensor_resolution = sensor_resolution
        self.sensor_width = sensor_width
        self.step = 5
        self.turn_angle = pi / 12

        self.max_eye_length = distance(XYPoint(0, 0), XYPoint(width, height))
        self.eye_length = min(sensor_length, self.max_eye_length)


        eye_max_angle = self.sensor_width
        if n_sensors > 1:
            eye_step = (eye_max_angle) / (n_sensors - 1)
        else:
            eye_step = 0

        self.eyes = []
        # add the leftmost eye first, working our way from +sensor_width/2 to -sensor_width/2
        for i in range(0, n_sensors):
            if n_sensors == 1:
                angle = 0
            else:
                angle = eye_max_angle / 2 - eye_step * i
            self.eyes.append(
                Eye(xy, ship_angle, angle, eye_step, self.eye_length,  self.width, self.height))


    def can_eat(self, o) -> bool:
        if distance(o.xy, self.xy) < o.radius + 3:
            return True
        else:
            return False

    def turn(self, angle):
        self.angle = reduce_angle(self.angle + angle)


    def move_right(self) -> float:
        self.turn(-self.turn_angle)
        return self.move_forward()

    def move_left(self) -> float:
        self.turn(self.turn_angle)
        return self.move_forward()

    def move_forward(self) -> float:
        x = self.xy.x + self.step * cos(self.angle)
        y = self.xy.y + self.step * sin(self.angle)

        bounce = False
        if x < 0:
            x = 0
            self.angle = reduce_angle(-self.angle + pi)
            bounce = True

        if x > self.width:
            x = self.width
            self.angle = reduce_angle(-self.angle + pi)
            bounce = True

        if y < 0:
            y = 0
            self.angle = reduce_angle(-self.angle)
            bounce = True

        if y > self.height:
            y = self.height
            self.angle = reduce_angle(-self.angle)
            bounce = True

        self.xy = XYPoint(x, y)
        for e in self.eyes:
            e.move(self.xy, self.angle)

        if (bounce):
            return -1.0
        else:
            return 0.0
