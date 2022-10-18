# project imports
from rectangle import Rectangle


class QuadTree:

    def __init__(self,
                 boundary,
                 n: int):
        self.boundary = boundary
        self.capacity = n
        self.movels = []
        self.divided = False

        self.top_left = None
        self.top_right = None
        self.bottom_left = None
        self.bottom_right = None

    def clear(self):
        self.divided = False
        self.movels = []

    def insert(self,
               movel):
        # If the hagaboob is beyond the the boundaries of the arena than it can't be inserted into any sub-rectangle
        if not self.boundary.contains(movel=movel):
            return False

        # If the current quadtree is divided already try to add the hagaboob to one the 4 sub-areas
        if self.divided:
            if self.top_left.insert(movel):
                return True
            elif self.top_right.insert(movel):
                return True
            elif self.bottom_left.insert(movel):
                return True
            elif self.bottom_right.insert(movel):
                return True
        else:
            self.movels.append(movel)

        if len(self.movels) > self.capacity:
            self._subdivide()

            for m in self.movels:
                if self.top_left.insert(m):
                    continue
                elif self.top_right.insert(m):
                    continue
                elif self.bottom_left.insert(m):
                    continue
                elif self.bottom_right.insert(m):
                    continue

            self.movels = []
            return True

    def query(self,
              area,
              highlight=False):
        found = []
        self.highlighted = False

        if self.boundary.intersects(area):
            if self.divided:
                found_tl = self.top_left.query(area, highlight)
                found_tr = self.top_right.query(area, highlight)
                found_bl = self.bottom_left.query(area, highlight)
                found_br = self.bottom_right.query(area, highlight)
                found = found.extend([found_tl, found_tr, found_bl, found_br])
            else:
                if highlight:
                    self.highlighted = True
                found.extend([m for m in self.movels if area.contains(m)])
        return found

    def _subdivide(self):
        x = self.boundary.x
        y = self.boundary.y
        w = self.boundary.w
        h = self.boundary.h

        # rectangle coordinates define its center!!!
        rect_tl = Rectangle(x - w / 4, y - h / 4, w / 2, h / 2)
        self.top_left = QuadTree(rect_tl, self.capacity)

        rect_tr = Rectangle(x + w / 4, y - h / 4, w / 2, h / 2)
        self.top_right = QuadTree(rect_tr, self.capacity)

        rect_bl = Rectangle(x - w / 4, y + h / 4, w / 2, h / 2)
        self.bottom_left = QuadTree(rect_bl, self.capacity)

        rect_br = Rectangle(x + w / 4, y + h / 4, w / 2, h / 2)
        self.bottom_right = QuadTree(rect_br, self.capacity)

        self.divided = True
