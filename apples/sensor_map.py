from ship import Ship
import numpy as np
from math import pi
from mathutils import reduce_angle, distance


class SensorMap(object):
    def __init__(self, ship: Ship):
        self.resolution = ship.sensor_resolution
        self.ship = ship
        self.n_sensors = ship.n_sensors
        self.screen: np.ndarray = np.zeros((3 * ship.n_sensors, ship.sensor_resolution))
        self.size = len(self.screen.flatten())
        self.sensor_width = ship.sensor_width
        self.sensor_resolution = ship.sensor_resolution

    def as_input(self) -> np.ndarray:
        return self.screen.reshape((1, self.size)).copy()

    def update(self, all_apples: []) -> np.ndarray:
        ship = self.ship

        self.screen.fill(0.0)

        ship_angle = self.ship.angle
        for a in all_apples:
            # the angle from the ship-xy to the object-xy
            object_angle = (a.xy - self.ship.xy).angle()
            # this is the angle from the left edge of the sensor field (ship_angle + sensor_width / 2)
            # to the object.  This is stated as a positive angle in the clockwise direction.
            net_angle = reduce_angle((ship_angle + self.sensor_width / 2) - object_angle)

            d = distance(a.xy, self.ship.xy)

            # use net_angle instead of sensor_index > 0 because int(-.5) = 0
            if (net_angle >= 0 and net_angle < self.sensor_width and d < ship.eye_length):
                # take the fraction of the distance through the sensor field and translate that into an integer
                # corresponding to the nearest sensor
                sensor_idx = int(net_angle / self.sensor_width * self.n_sensors)
                col = min(self.sensor_resolution - 1, int(d / ship.eye_length * self.sensor_resolution + .5))
                if a.ripe():
                    self.screen[sensor_idx * 3 + 0, col] += 1
                else:
                    self.screen[sensor_idx * 3 + 1, col] += 1

        for idx, eye in enumerate(ship.eyes):
            n_blocked = int((eye.length - eye.actual_length) / eye.length * self.resolution + .5)
            for d in range(n_blocked):
                self.screen[idx * 3 + 2, self.resolution - d - 1] = 1

        return self.screen

    def print(self):
        print("screen.shape: ", self.screen.shape[0], self.screen.shape[1])
        for r in range(self.screen.shape[0]):
            for c in range(self.screen.shape[1]):
                print("%.0f " % self.screen[r, c], end='')
            print()

