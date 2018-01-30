import numpy as np
from xypoint import XYPoint
import arcade as a


class ArrayVis(object):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def draw(self, data: np.ndarray, lower_left: XYPoint, row_colors):
        (rows, cols) = data.shape

        self.x_cell_size = self.width / cols
        self.y_cell_size = self.height / rows

        ll_centered = XYPoint(lower_left.x + self.x_cell_size / 2, lower_left.y + self.y_cell_size / 2)

        #a.draw_rectangle_filled(
            #lower_left.x + self.width / 2,
            #lower_left.y + self.height / 2,
            #self.width,
            #self.height,
            #a.color.WHITE)

        for r in range(rows):
            for c in range(cols):
                value = data[r, c]
                center = ll_centered + XYPoint(c * self.x_cell_size, (rows - 1 - r) * self.y_cell_size)

                base_color = row_colors[r % len(row_colors)]

                if value > 1:
                    value = 1
                elif value < 0:
                    value = 0

                if (value > .05):
                    cc = (int(base_color[0] * value), int(base_color[1] * value), int(base_color[2] * value))

                    a.draw_rectangle_filled(center.x,
                                            center.y,
                                            self.x_cell_size,
                                            self.y_cell_size,
                                            cc)

