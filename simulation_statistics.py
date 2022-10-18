# library imports
import os
import math
import json
import numpy as np

# project imports
from vector import Vector
import matplotlib.pyplot as plt


class SimulationStatistics:
    """

    """

    def __init__(self):
        self.angles = []
        self.stds = []
        self.hits = []
        self.vels = []
        self.vars = []
        self.arr_x = []
        self.arr_y = []
        self.r = 0

    def get_angles(self,
                   movels=None):
        self.angles = []
        self.arr_x = []
        self.arr_y = []
        if movels is not None and len(movels) > 0:
            for m in movels:
                angle = m.vel.heading()
                self.arr_x.append(math.cos(angle))
                self.arr_y.append(math.sin(angle))
                self.angles.append(angle)

    def add_std(self):
        sum_x = sum(self.arr_x)
        sum_y = sum(self.arr_y)
        avg_x = sum_x / len(self.arr_x) if len(self.arr_x) > 0 else 0
        avg_y = sum_y / len(self.arr_y) if len(self.arr_y) > 0 else 0
        r_mean = math.sqrt(avg_x * avg_x + avg_y * avg_y)
        self.vars.append(1 - r_mean)
        std = math.sqrt(-2 * math.log(r_mean))
        self.stds.append(std)

    def add_vels(self, hagabubim):

        # add average vel
        self.vels.append(Vector.mean(vectors=[hagabub.vel for hagabub in hagabubim]))

    def add_hits(self, hagabubim):
        # add average vel
        self.hits.append(np.mean([1 if hagabub.is_colloid else 0 for hagabub in hagabubim]))

    def download(self,
                 path: str):
        # make sure we have this folder
        try:
            os.mkdir(os.path.dirname(path))
        except Exception as error:
            pass

        with open(path, "w") as answer_file:
            json.dump({"circular stds": self.stds,
                       "hit_rate": self.hits},
                      answer_file)

        plt.plot(list(range(1, len(self.stds) + 1)),
                 self.stds,
                 "-o",
                 color="blue",
                 markersize=2,
                 label="Circular STDS")

        plt.plot(list(range(1, len(self.hits) + 1)),
                 self.hits,
                 "-o",
                 color="red",
                 markersize=2,
                 label="Hit rate")
        plt.xlabel("Simulation step [1]")
        plt.ylabel("Simulation Signals [1]")
        plt.legend()
        plt.savefig(path.replace(".json", ".png"))
        plt.close()

    def download2(self,
                  path: str):
        # make sure we have this folder
        try:
            os.mkdir(path)
        except Exception as error:
            pass

        with open(path, "w") as answer_file:
            json.dump(self.vars, answer_file)

    def __repr__(self):
        return "<SimulationStatistics>"

    def __str__(self):
        return "<SimulationStatistics: {} samples>".format(len(self.stds))
