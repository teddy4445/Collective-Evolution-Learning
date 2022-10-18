# library imports
import math


class Rectangle:

    def __init__(self,
                 x: float,
                 y: float,
                 w: float,
                 h: float):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def contains(self,
                 movel):
        dist_x = abs(self.x - movel.loc.x)
        dist_y = abs(self.y - movel.loc.y)
        return dist_x < self.w / 2 and dist_y < self.h / 2

    def intersects(self,
                   area):
        test_x = area.x
        test_y = area.y
        if area.x < self.x - self.w / 2:
            test_x = self.x - self.w / 2  # left edge
        elif area.x > self.x + self.w / 2:
            test_x = self.x + self.w / 2  # right edge
        if area.y < self.y - self.h / 2:
            test_y = self.x - self.h / 2  # top edge
        elif area.y > self.x + self.h / 2:
            test_y = self.x + self.h / 2  # bottom edge

        dist_x = area.x - test_x
        dist_y = area.y - test_y
        dist = math.sqrt(dist_x * dist_x + dist_y * dist_y)
        return dist <= area.r

    def __repr__(self):
        return "Rectangle"

    def __str__(self):
        return "<Rectangle | ({},{}), h X w = {} X {}>".format(self.x,
                                                               self.y,
                                                               self.w,
                                                               self.h)
