from xypoint import XYPoint


class Apple(object):
    def __init__(self,  xy: XYPoint, radius: int, red: bool):
        self.xy = xy
        self.radius = radius
        self.red = red

    def ripe(self) -> bool:
        if self.red:
            return True
        else:
            return False

