# library imports
import math
import random

# project imports
from settings import *
from circle import Circle
from vector import Vector, create_vector
from separating_axis_theorem import separating_axis_theorem


def mod(a, b):
    return ((a % b) + b) % b


class Hagabub:
    """
    A single agent in the swarm simulation
    """

    def __init__(self,
                 index,
                 brain,
                 length,
                 wid,
                 vision_radius,
                 w_screen,
                 h_screen,
                 quadtree,
                 max_speed,
                 cheat_factor,
                 loc: Vector = None,
                 vel: Vector = None):
        # logic members
        self.index = index
        self.brain = brain

        # physics settings
        self.maxSpeed = max_speed
        self.len = length
        self.wid = wid
        self.d = math.sqrt(math.pow(self.len, 2) + math.pow(self.wid, 2))

        # physics
        if loc is None:
            self.loc = create_vector(random.randint(0, w_screen), random.randint(0, h_screen))
        else:
            self.loc = loc
        if vel is None:
            self.vel = create_vector(random.random() * 2 * self.maxSpeed - self.maxSpeed,
                                     random.random() * 2 * self.maxSpeed - self.maxSpeed)
        else:
            self.vel = vel

        self.orientation = self.vel.heading()
        self.is_colloid = False

        self.corners = self._calc_corners()
        self.vision = Circle(self.loc.x,
                             self.loc.y,
                             vision_radius)

        # quad-tree for vision
        self.qt = quadtree

        # decision-support members
        self.neighbors = []
        self.alphas = []
        self.prev_alphas = []
        self.prevLoc = create_vector(random.randint(0, w_screen), random.randint(0, h_screen))
        self.prevVel = create_vector(random.randint(-self.maxSpeed, self.maxSpeed),
                                     random.randint(-self.maxSpeed, self.maxSpeed))
        self.prevCorners = [[random.randint(0, w_screen), random.randint(0, h_screen)],
                            [random.randint(0, w_screen), random.randint(0, h_screen)],
                            [random.randint(0, w_screen), random.randint(0, h_screen)],
                            [random.randint(0, w_screen), random.randint(0, h_screen)]]

        self.edges = False
        self.sides = False

        # simulation global members
        self.w_screen = w_screen
        self.h_screen = h_screen
        self.step_index = 0
        self.cheat_factor = cheat_factor

        # memory members
        self.wait_turns = 0
        self.last_vel = Vector.zero()

    # main logic #

    def update(self):
        """
        Updates some hagabub parameters and enforces boundary conditions
        :return:
        """
        self.step_index += 1
        # check if we wait after collision
        if self.wait_turns == 0:
            state = self.gather_state()
            try:
                action = self.brain.predict_max(neighbors=state)
            except:
                action = 0
            # recall last vel
            self.last_vel = self.vel.copy()
            # update location
            if (self.step_index % self.cheat_factor) == 0:
                self._update_vel_exact()
            else:
                self._update_vel(action=action)
            self._update_loc()
            self.vision.update(self.loc.x,
                               self.loc.y)
        else:
            self.is_colloid = False
            self.wait_turns -= 1
            # first time, just go to the place you went before
            if self.wait_turns == 0:
                self.vel = self.last_vel.copy().mult(5)
                self._update_loc()
                self.vision.update(self.loc.x,
                                   self.loc.y)

    def gather_state(self):
        """
        :return: state of hagabub containing alpha,beta,alpha_dot,beta_dot for each neighbor
        """
        state_lst = [round(self.vel.x, MODEL_PARAMETER_DISCRIMINATION), round(self.vel.y, MODEL_PARAMETER_DISCRIMINATION)]
        self.neighbors = self.qt.query(self.vision)
        hagabub_count = 0
        for hagabub_index, neigh in enumerate(self.neighbors):
            if neigh.index != self.index and hagabub_count < MAX_NEIGHBORS:
                alpha = round(self._subtended_angle(neigh, True) / 2, MODEL_PARAMETER_DISCRIMINATION)
                beta = round(self._bearing(neigh), MODEL_PARAMETER_DISCRIMINATION)
                alpha_dot = round((self._subtended_angle(neigh, True) - self._subtended_angle(neigh, False)) / 2, MODEL_PARAMETER_DISCRIMINATION)
                beta_dot = round(self._delta_bearing(neigh), MODEL_PARAMETER_DISCRIMINATION)

                state_lst.extend([alpha, alpha_dot, beta, beta_dot])

                hagabub_count += 1
        return state_lst

    def _update_vel_exact(self):
        # if collide, stop them for one step
        if self._is_collision():
            self.wait_turns = 1 + round(random.random() * (WAIT_TURNS_MAX-1))
            self.vel = Vector.zero()
            return
        self.is_colloid = False
        sumVel = Vector(0, 0)
        for neighbor in self.neighbors:
            sumVel.add(neighbor.vel)
        exactVel = sumVel.normalize(len(self.neighbors))
        self.vel = self.vel.mult(1 - ACTION_INFLUENCE_W).add(other_vector=exactVel.mult(ACTION_INFLUENCE_W)).normalize(self.maxSpeed)

    def _update_vel(self,
                    action: int):
        """
        :param new_vel:
        :return:
        """
        # if collide, stop them for one step
        if self._is_collision():
            self.wait_turns = 1 + round(random.random() * (WAIT_TURNS_MAX-1))
            self.vel = Vector.zero()
            return
        # find the right action
        self.is_colloid = False
        turn = 0
        if action == 0:
            return
        elif action < 7:
            turn = action * 15
        elif action < 13:
            turn = (action-6) * -15

        self.vel = self.vel.mult(1-ACTION_INFLUENCE_W).add(other_vector=self.vel.copy().mult(ACTION_INFLUENCE_W).rotate(turn)).normalize(self.maxSpeed)

    def _update_loc(self):
        # update location as physical engine
        self.loc = self.loc.add(self.vel)
        # update location inside the map if the hagabub is outside the map
        if self.loc.x < 0:
            self.loc.x += W_SCREEN
        if self.loc.x > W_SCREEN:
            self.loc.x -= W_SCREEN
        if self.loc.y < 0:
            self.loc.y += H_SCREEN
        if self.loc.y > H_SCREEN:
            self.loc.y -= H_SCREEN
        # update corners
        self.corners = self._calc_corners()

    # end - main logic #

    # vision logic #

    def _calc_corners(self):
        """
        Calculates the corners locations of the unit
        :return: array of corners coordinates
        """
        v = self.vel.copy().unit_vector()
        l = self.len
        w = self.wid
        a = Vector(l * v.x - w * v.y, l * v.y + w * v.x).mult(0.5)
        b = Vector(l * v.x + w * v.y, l * v.y - w * v.x).mult(0.5)
        c = Vector(-l * v.x - w * v.y, -l * v.y + w * v.x).mult(0.5)
        d = Vector(-l * v.x + w * v.y, -l * v.y - w * v.x).mult(0.5)
        corners = [a, b, c, d]

        if a.x == 0 and a.y == 0:
            vx = math.cos(self.orientation)
            vy = math.sin(self.orientation)
            a = Vector(l * vx - w * vy, l * vy + w * vx).mult(0.5)
            b = Vector(l * vx + w * vy, l * vy - w * vx).mult(0.5)
            c = Vector(-l * vx - w * vy, -l * vy + w * vx).mult(0.5)
            d = Vector(-l * vx + w * vy, -l * vy - w * vx).mult(0.5)
            corners = [a, b, c, d]
        else:
            self.orientation = self.vel.heading()

        return [corner.add(self.loc) for corner in corners]

    def _periodSub(self,
                   v2,
                   v1):
        """
        Special substitution that accounts the periodicity of the arena
        :param v2:
        :param v1:
        :return:
        """
        try:
            v2 = Vector.from_list(v2)
        except:
            pass
        try:
            v1 = Vector.from_list(v1)
        except:
            pass
        x = mod(v2.x - v1.x + self.w_screen / 2, self.w_screen) - self.w_screen / 2
        y = mod(v2.y - v1.y + self.h_screen / 2, self.h_screen) - self.h_screen / 2
        return create_vector(x, y)

    def _subtended_angle(self,
                         n,
                         time):
        """
        Returns the substituting angle of the neighbor as seen on our circular lens
        :param n: neighbor
        :param time: True for now and False for before
        :return: alpha
        """
        corners = n.corners
        los = self._periodSub(n.loc, self.loc)
        if not time:
            corners = n.prevCorners
            los = self._periodSub(n.prevLoc, self.prevLoc)
        diagonal1 = self._periodSub(corners[0], corners[2])
        diagonal2 = self._periodSub(corners[1], corners[3])
        if abs(diagonal1.angle_between(los)) > abs(diagonal2.angle_between(los)):
            v1 = self._periodSub(corners[0], self.loc)
            v2 = self._periodSub(corners[2], self.loc)
        else:
            v1 = self._periodSub(corners[1], self.loc)
            v2 = self._periodSub(corners[3], self.loc)
        return abs(v1.angle_between(v2))

    def _bearing(self,
                 n):
        """
        The relative angle between the ego-direction of self and a neighbor - the bearing
        :param n: neighbor index
        :return: beta
        """
        los = n.loc.sub(self.loc)
        rotation_dir = self.vel.cross(los, is_3d=False)[2]
        if rotation_dir > 0:
            return self.vel.angle_between(los)
        else:
            return self.vel.angle_between(los) * -1

    def _delta_bearing(self,
                       n):
        """
        Calculates the difference between two consecutive time frames of beta
        :param n: neighbor
        :return: beta-dot
        """
        # TODO: fix the cyclic periodicity behind self
        prevLos = self._periodSub(n.prevLoc, self.prevLoc)
        try:
            rotation_dir = self.prevVel.cross(prevLos, is_3d=False)[2]
            prevBearing = self.prevVel.angle_between(prevLos)
            if rotation_dir < 0:
                prevBearing *= -1
        except:
            prevBearing = self.prevVel.angle_between(prevLos)

        beta_now = self._bearing(n)
        delta = beta_now - prevBearing
        tuned = delta - math.pi * 2 * round(delta / math.pi * 2)
        if tuned > 1:
            return 0
        return tuned

    # end - vision logic #

    # collision logic #

    def _is_collision(self):
        """
        Check if we collide with one of the neighbors
        :return: the hagabub we hit
        """

        for neighbor in self.neighbors:
            if not Vector.is_zero(neighbor.vel) and self._is_pairwaise_collision(other_hagabub=neighbor) and self.loc.sub(other_vector=neighbor.loc).mag() < 3 * max(self.len, self.wid):
                self.is_colloid = True
                return True
        self.is_colloid = False
        return False

    def _is_pairwaise_collision(self,
                                other_hagabub):
        """
        Compute either we collide with other hagabub or not
        :param other_hagabub: the hagabub to check with
        :return: true if collided, false otherwise
        """
        if other_hagabub is not self:
            return separating_axis_theorem(vertices_a=self.corners,
                                           vertices_b=other_hagabub.corners)

    # end - collision logic #

    # sim states functions #

    def get_state(self):
        return [self.loc.x, self.loc.y, self.vel.x, self.vel.y]

    # end - sim states functions #

    def __repr__(self):
        return "<Hagabub | at: {:.3f}X{:.3f}>".format(self.loc.x, self.loc.y)

    def __str__(self):
        return "<Hagabub | at: {:.3f}X{:.3f}, speed: [{:.3f}, {:.3f}]>".format(self.loc.x,
                                                                               self.loc.y,
                                                                               self.vel.x,
                                                                               self.vel.y)
