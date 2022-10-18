# library imports
import math


class Circle:

    def __init__(self,
                 x,
                 y,
                 r):
        self.x = x
        self.y = y
        self.r = r

    def update(self,
               x,
               y):
        self.x = x
        self.y = y

    def contains(self,
                 movel):
        dist_x = self.x - movel.loc.x
        dist_y = self.y - movel.loc.y
        dist = math.sqrt(dist_x * dist_x + dist_y * dist_y)
        return dist < self.r

    def contains_loc(self,
                     x,
                     y):
        dist_x = self.x - x
        dist_y = self.y - y
        dist = math.sqrt(dist_x * dist_x + dist_y * dist_y)
        return dist < self.r

    def __repr__(self):
        return "Circle"

    def __str__(self):
        return "<Circle | ({},{}), r={}>".format(self.x,
                                                 self.y,
                                                 self.r)
