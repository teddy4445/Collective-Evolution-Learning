# library imports
import time
import random
import numpy as np

# project import
from settings import *
from circle import Circle
from vector import Vector
from hagabub import Hagabub
from quad_tree import QuadTree
from rectangle import Rectangle
from model_samples_generator_ga import ModelSamplesGeneratorGeneticAlgorithm


class ModelSamplesGenerator:
    """
    This class generates samples for the model to learn
    """

    def __init__(self):
        pass

    @staticmethod
    def generate(samples: int,
                 max_speed: float,
                 see_radius: float,
                 max_neighbors: int,
                 is_test: bool = False):
        x = []
        y = []
        global_start_time = time.time()
        start_time = time.time()

        # generate random cases
        for sample in range(round(samples * (1 - GENERATOR_FLOCK_SAMPLES_RATE))):
            # just for debug
            if (sample % 10000) == 0:
                print("It took {} seconds to generate this 10000 samples".format(time.time() - start_time))
                start_time = time.time()
                print("Done {}/{} ({:.2f}%)".format(sample + 1, samples, 100 * (sample + 1) / samples))
            single_x, single_y = ModelSamplesGenerator._generate_single(max_speed=max_speed,
                                                                        see_radius=see_radius,
                                                                        max_neighbors=max_neighbors,
                                                                        is_test=is_test)
            x.append(single_x)
            y.append(single_y)

        # generate flocked samples
        for sample in range(round(samples * GENERATOR_FLOCK_SAMPLES_RATE)):
            # just for debug
            if (sample % 10000) == 0:
                print("It took {} seconds to generate this 10000 samples".format(time.time() - start_time))
                start_time = time.time()
                print("Done {}/{} ({:.2f}%)".format(sample + 1, samples, 100 * (sample + 1) / samples))
            single_x, single_y = ModelSamplesGenerator._generate_single_flocked(max_speed=max_speed,
                                                                                see_radius=see_radius,
                                                                                max_neighbors=max_neighbors)
            x.append(single_x)
            y.append(single_y)
        print("It took {} seconds to generate the entire {} dataset samples".format(time.time() - global_start_time, samples))
        return x, y

    @staticmethod
    def _generate_single(max_speed: float,
                         see_radius: float,
                         max_neighbors: int,
                         is_test: bool = False) -> tuple:
        global MODEL_TYPE
        global MODEL_PARAMETER_DISCRIMINATION
        # calc the 'x' columns #
        # this agent's vel
        this_vel = Vector.random().normalize(scalar=max_speed)
        x = []
        # generate random number of neighbors
        neighbors_count = round(random.random() * max_neighbors)
        neighbors = []
        see_circle = Circle(0, 0, see_radius)
        for neighbors_index in range(neighbors_count):
            delta_x = round(random.random() * see_radius)
            delta_y = round(random.random() * see_radius)
            while not see_circle.contains_loc(x=delta_x,
                                              y=delta_y):
                delta_x = round(random.random() * see_radius)
                delta_y = round(random.random() * see_radius)
            new_vel = Vector.random().normalize(scalar=max_speed)
            neighbors.extend([delta_x, delta_y, new_vel.x, new_vel.y])
        x.extend(ModelSamplesGenerator.state_neighbors(my_vel=this_vel,
                                                       neighbors=neighbors))

        left_places = max_neighbors - neighbors_count
        for neighbors_index in range(left_places):
            x.extend([0, 0, 0, 0])
        x = [round(val, MODEL_PARAMETER_DISCRIMINATION) for val in x]

        if MODEL_TYPE == "random" and not is_test:
            return x, random.randint(0, 12)
        elif MODEL_TYPE == "cm" or is_test:
            # calc the 'y' column #
            y = ModelSamplesGenerator.right_action(my_vel=this_vel,
                                                   neighbors=neighbors)
            return x, y
        elif not is_test:
            # calc the 'y' column #
            y = ModelSamplesGenerator.ga_action(my_vel=this_vel,
                                                neighbors=neighbors)
            return x, y

    @staticmethod
    def _generate_single_flocked(max_speed: float,
                                 see_radius: float,
                                 max_neighbors: int) -> tuple:
        # TODO: change the random.random() function to be based on the evaluation phase
        # calc the 'x' columns #
        # this agent's vel
        this_vel = Vector.random().normalize(scalar=max_speed)
        x = []
        # generate random number of neighbors
        neighbors_count = round(random.random() * max_neighbors)
        if neighbors_count < 2:
            neighbors_count = 2
        # pick a random direction for the neighbors
        flock_vel = Vector.random().normalize(scalar=max_speed)
        # generate neighbors
        neighbors = []
        see_circle = Circle(0, 0, see_radius)
        for neighbors_index in range(neighbors_count):
            delta_x = round(random.random() * see_radius)
            delta_y = round(random.random() * see_radius)
            while not see_circle.contains_loc(x=delta_x,
                                              y=delta_y):
                delta_x = round(random.random() * see_radius)
                delta_y = round(random.random() * see_radius)
            new_vel = flock_vel.add(other_vector=Vector.random().normalize(1/(GENERATOR_FLOCK_VEL_NOISE_FACTOR*max_speed)))
            neighbors.extend([delta_x, delta_y, new_vel.x, new_vel.y])
        x.extend(ModelSamplesGenerator.state_neighbors(my_vel=this_vel,
                                                       neighbors=neighbors))

        left_places = max_neighbors - neighbors_count
        for neighbors_index in range(left_places):
            x.extend([0, 0, 0, 0])

        # calc the 'y' column #
        y = ModelSamplesGenerator.right_action(my_vel=this_vel,
                                               neighbors=neighbors)

        return x, y

    @staticmethod
    def state_neighbors(my_vel: Vector,
                        neighbors: list) -> list:
        """
        :param my_vel:
        :param neighbors: a list of the form list of 4 values: [delta_loc_x, delta_loc_y, abslute_vel_x, absolute_vel_y]
        :return: a new vector of the neighbors with [alpha, alpha_dot, beta, beta_dot]
        """
        # neighbor size declare
        NEIGHBOR_SIZE = 4
        # split to a list of each neighbor
        each_neighbor = [tuple(neighbors[start_index:start_index + NEIGHBOR_SIZE]) for start_index in
                         range(0, len(neighbors), NEIGHBOR_SIZE)]
        # create qt
        qt = QuadTree(boundary=Rectangle(W_SCREEN / 2, H_SCREEN / 2, W_SCREEN, H_SCREEN),
                      n=len(each_neighbor) + 1)
        # create the list of hgabubs
        hagabub_pop = [Hagabub(index=0,
                               loc=Vector(W_SCREEN / 2, H_SCREEN / 2),
                               vel=my_vel,
                               length=HAGABUB_LENGTH,
                               wid=HAGABUB_WIDTH,
                               brain=None,
                               quadtree=qt,
                               h_screen=H_SCREEN,
                               w_screen=W_SCREEN,
                               vision_radius=VISION_RADUIS,
                               max_speed=VISION_RADUIS,
                               cheat_factor=MAX_STEPS+1)]
        for index, neigh in enumerate(each_neighbor):
            hagabub_pop.append(Hagabub(index=index + 1,
                                       loc=Vector(x=neigh[0] + W_SCREEN / 2,
                                                  y=neigh[1] + H_SCREEN / 2),
                                       vel=Vector(x=neigh[2],
                                                  y=neigh[3]),
                                       brain=None,
                                       quadtree=qt,
                                       length=HAGABUB_LENGTH,
                                       wid=HAGABUB_WIDTH,
                                       h_screen=H_SCREEN,
                                       w_screen=W_SCREEN,
                                       vision_radius=VISION_RADUIS,
                                       max_speed=MAX_SPEED,
                                       cheat_factor=MAX_STEPS+1))
        # insert the virtual population of hagabubim in the QT
        [qt.insert(movel=movel) for movel in hagabub_pop]
        hagabub_pop[0].qt = qt
        return hagabub_pop[0].gather_state()

    @staticmethod
    def right_action(my_vel: Vector,
                     neighbors: list) -> int:
        """
        :param my_vel:
        :param neighbors: a list of the form list of 4 values: [delta_loc_x, delta_loc_y, abslute_vel_x, absolute_vel_y]
        :return: a list [vel_x, vel_y] that describe the ideal velocity the hagabub should take given these neighbors
        """
        # neighbor size declare
        NEIGHBOR_SIZE = 4
        # split to a list of each neighbor
        each_neighbor = [tuple(neighbors[start_index:start_index + NEIGHBOR_SIZE]) for start_index in
                         range(0, len(neighbors), NEIGHBOR_SIZE)]
        if not each_neighbor:
            # No neighbors present - keep
            return 0

        sum_vel_x, sum_vel_y = 0, 0
        for neigh in each_neighbor:
            sum_vel_x += neigh[2]
            sum_vel_y += neigh[3]
        # avg and transform vel to relative velocity
        avg_vel = Vector(sum_vel_x / len(each_neighbor), sum_vel_y / len(each_neighbor)).sub(my_vel)
        return ModelSamplesGenerator.pick_action_from_vector(vector=avg_vel)

    @staticmethod
    def ga_action(my_vel: Vector,
                  neighbors: list) -> int:
        """
        :param my_vel:
        :param neighbors: a list of the form list of 4 values: [delta_loc_x, delta_loc_y, abslute_vel_x, absolute_vel_y]
        :return: a list [vel_x, vel_y] that describe the ideal velocity the hagabub should take given these neighbors
        """
        vector = ModelSamplesGeneratorGeneticAlgorithm.run(start_guess=Vector.random(),
                                                           my_vel=my_vel,
                                                           neighbors=neighbors)
        return ModelSamplesGenerator.pick_action_from_vector(vector=vector)

    @staticmethod
    def pick_action_from_vector(vector):
        # convert the avg vel vector to angle in degrees
        angle = vector.heading() * 180 / np.pi
        if 82.5 < angle <= 97.5:
            # keep
            return 0
        if 97.5 < angle <= 112.5:
            # turn left 15
            return 1
        if 112.5 < angle <= 127.5:
            # turn left 30
            return 2
        if 127.5 < angle <= 142.5:
            # left 45
            return 3
        if 142.5 < angle <= 157.5:
            # left 60
            return 4
        if 157.5 < angle <= 172.5:
            # left 75
            return 5
        if 172.5 < angle or angle <= -90:
            # left 90
            return 6
        if 67.5 < angle <= 82.5:
            # turn right 15
            return 7
        if 52.5 < angle <= 67.5:
            # right 30
            return 8
        if 37.5 < angle <= 52.5:
            # right 45
            return 9
        if 22.5 < angle <= 37.5:
            # right 60
            return 10
        if 7.5 < angle <= 22.5:
            # right 75
            return 11
        if angle <= 7.5 or angle > -90:
            # right 90
            return 12
        else:
            raise ValueError('Something wrong with avg angle calculation')
