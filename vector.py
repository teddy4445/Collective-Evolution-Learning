# library imports
import math
from math import cos, sin
import random
import numpy as np


def create_vector(x=0,
                  y=0):
    return Vector(x,
                  y)


class Vector:
    """
    Algebraic vector operation class
    """

    def __init__(self,
                 x,
                 y):
        self.x = x
        self.y = y

    @staticmethod
    def zero():
        return Vector(0, 0)

    @staticmethod
    def is_zero(vector):
        return vector.x == 0 and vector.y == 0

    @staticmethod
    def random():
        return Vector(random.random() * 2 - 1,
                      random.random() * 2 - 1)

    @staticmethod
    def turn_right():
        return Vector(1, 0)

    @staticmethod
    def turn_left():
        return Vector(-1, 0)

    @staticmethod
    def turn_up():
        return Vector(0, 1)

    @staticmethod
    def turn_down():
        return Vector(0, -1)

    @staticmethod
    def from_list(values: list):
        return Vector(x=values[0],
                      y=values[1])

    @staticmethod
    def mean(vectors: list):
        answer = Vector.zero()
        for vector in vectors:
            answer = answer.add(other_vector=vector)
        return answer

    def to_np_array(self):
        return np.asarray([self.x, self.y])

    def add(self, other_vector):
        return Vector(self.x + other_vector.x,
                      self.y + other_vector.y)

    def sub(self, other_vector):
        return Vector(self.x - other_vector.x,
                      self.y - other_vector.y)

    def mult(self, scalar):
        """
        :param scalar:
        :return: Vector multiplied by the scalar
        """
        return Vector(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        """
        Returns the vector after division by a scalar
        :param scalar:
        :return:
        """
        return Vector(self.x / scalar, self.y / scalar) if scalar != 0 else Vector(0, 0)

    def unit_vector(self):
        """ Returns the unit vector of the vector.  """
        return self.copy() / np.linalg.norm((self.x, self.y))

    def normalize(self,
                  scalar: float = 1):
        """ Returns the unit vector of the vector.  """
        return self.unit_vector().mult(scalar=scalar)

    def mag(self):
        """
        Returns the magnitude of the vector
        :return:
        """
        return math.sqrt(self.x * self.x + self.y * self.y)

    def set_mag(self, newMag):
        """
        Sets the vector magnitude to be newMag
        :param newMag:
        :return:
        """
        ratio = newMag / self.mag()
        self.x *= ratio
        self.y *= ratio
        return self

    def limit(self, maxMag, is_upper):
        """
        Limits the maximal magnitude of a vector to maxMag
        :param maxMag:
        :return:
        """
        if self.mag() > maxMag and is_upper:
            return self.set_mag(maxMag)
        elif self.mag() < maxMag and not is_upper:
            return self.set_mag(maxMag)
        return self

    def angle_between(self, v2):
        """
        Returns the angle in radians between vectors 'v1' and 'v2'::

        """
        v1_u = self.unit_vector()
        v2_u = v2.unit_vector()
        return np.arccos(np.clip(np.dot(v1_u.to_np_array(), v2_u.to_np_array()), -1.0, 1.0))

    def angle_x_axis(self):
        return self.angle_between(Vector.turn_right())

    def cross(self,
              other_vector,
              is_3d: bool = True):
        """
        Returns the perpendicular vector to both given vectors - the cross product
        :param other_vector:
        :return:
        """
        if is_3d:
            return np.cross([self.x, self.y], [other_vector.x, other_vector.y])
        else:
            return np.cross([self.x, self.y, 0], [other_vector.x, other_vector.y, 0])

    def heading(self):
        """
        Returns the heading angle of the vector
        :return:
        """
        return math.atan2(self.y, self.x)

    def perpendicular(self,
                      is_clock_wise: bool = True):
        if is_clock_wise:
            return Vector(x=self.y,
                          y=-1 * self.x)
        else:
            return Vector(x=-1 * self.y,
                          y=self.x)

    def rotate(self,
               angle: float):
        vec = self.to_np_array()
        theta = np.deg2rad(angle)
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        vec_rotated = np.dot(rot, vec)
        return Vector(x=vec_rotated[0],
                      y=vec_rotated[1])

    def copy(self):
        """
        :return: copy instance of the original vector
        """
        return Vector(self.x, self.y)

    def __repr__(self):
        return "<Vector: [{:.3f}, {:.3f}]>".format(self.x,
                                                   self.y)

    def __str__(self):
        return "[{:.3f}, {:.3f}]".format(self.x, self.y)

