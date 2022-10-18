# library imports
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# project imports
from settings import *
from hagabub import Hagabub
from quad_tree import QuadTree
from rectangle import Rectangle
from simulation_statistics import SimulationStatistics


class Simulator:
    """

    """

    # end - loggers types #

    def __init__(self,
                 hagabub_model,
                 answer_folder: str,
                 cheat_factor: float = MAX_STEPS + 1,
                 max_steps: int = MAX_STEPS,
                 w_screen: int = W_SCREEN,
                 h_screen: int = H_SCREEN,
                 population_size: int = POPULATION_SIZE,
                 hagabub_length: float = HAGABUB_LENGTH,
                 hagabub_width: float = HAGABUB_WIDTH,
                 vision_radius: float = VISION_RADUIS,
                 max_speed: float = MAX_SPEED,
                 show_visual: bool = False):
        # init simulation stats
        self.stats = SimulationStatistics()
        # calc the agents on the simulation and some vision info
        boundry = Rectangle(w_screen / 2, h_screen / 2, w_screen, h_screen)
        self.qt = QuadTree(boundry, population_size)
        self.hagabubim = [Hagabub(index=i,
                                  brain=hagabub_model,
                                  length=hagabub_length,
                                  wid=hagabub_width,
                                  vision_radius=vision_radius,
                                  w_screen=w_screen,
                                  h_screen=h_screen,
                                  quadtree=self.qt,
                                  cheat_factor=cheat_factor,
                                  max_speed=max_speed)
                          for i in range(population_size)]

        [self.qt.insert(movel=movel) for movel in self.hagabubim]

        # used for modify simulation logic
        self.max_steps = max_steps
        self.show_visual = show_visual

        # answer_old members
        self.answer_folder = answer_folder

        # technical members
        self.step_index = 0
        self.display_states = []

        # technical for visual
        self.w_screen = w_screen
        self.h_screen = h_screen

    def run(self):
        # run the number of steps needed
        for step in range(self.max_steps):
            self.step()
            if not self.show_visual:
                print("Simulation at step: {} with circular std = {:.3f} and average vel = {} (|V| = {:.3f})".format(self.step_index, self.stats.stds[-1], self.stats.vels[-1], self.stats.vels[-1].mag()))
        # save results in the end
        self.stats.download(path=os.path.join(self.answer_folder, "answer_{}_final_{}.json".format(self.step_index,
                                                                                                   datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))))

    def step(self):
        # count this step
        self.step_index += 1

        # update the qt with the agents
        self.qt.capacity = len(self.hagabubim)
        self.qt.clear()
        [self.qt.insert(movel) for movel in self.hagabubim]

        # update the agents locations
        [movel.update() for movel in self.hagabubim]

        # show visual if requested
        if self.show_visual and (self.step_index % 1) == 0:
            self.visual()

        # update states
        self.stats.get_angles(movels=self.hagabubim)
        self.stats.add_std()
        self.stats.add_hits(hagabubim=self.hagabubim)
        self.stats.add_vels(hagabubim=self.hagabubim)

        # add the data to print later
        self.display_states.append(self.get_state())

    def get_state(self):
        return [movel.get_state() for movel in self.hagabubim]

    def visual(self,
               need_save: bool = False):
        # hagabub place and vector
        for hagabub_index, hagabub in enumerate(self.hagabubim):
            corners = hagabub.corners
            corners = sorted(corners, key=lambda corner: corner.x)
            keep = corners[2]
            corners[2] = corners[3]
            corners[3] = keep
            corners.append(corners[0])
            if len(hagabub.neighbors) == 1:
                plt.plot([corner.x for corner in corners],
                         [corner.y for corner in corners],
                         "-",
                         marker='',
                         c='black' if not hagabub.is_colloid else 'blue')
            else:
                plt.plot([corner.x for corner in corners],
                         [corner.y for corner in corners],
                         "--",
                         marker='o',
                         markersize=2,
                         c='black' if not hagabub.is_colloid else 'blue')
            show_vel = hagabub.vel.unit_vector().mult(10)
            plt.plot([hagabub.loc.x, hagabub.loc.x + show_vel.x],
                     [hagabub.loc.y, hagabub.loc.y + show_vel.y],
                     "--",
                     alpha=0,
                     c='red')
            plt.text(max([corner.x for corner in corners]) + 5,
                     max([corner.y for corner in corners]) + 5,
                     "{} (N={})".format(hagabub.index, len(hagabub.neighbors) - 1),
                     size=8)
            vision_circle = Circle((hagabub.loc.x, hagabub.loc.y),
                                   hagabub.vision.r,
                                   facecolor='none',
                                   edgecolor="black",
                                   linewidth=1,
                                   alpha=0.1)
            plt.gca().add_patch(vision_circle)

        plt.text(y=self.h_screen + 20,
                 x=0,
                 s="Step #{}: Circular STD: {:.3f}, |E[vel]| = {:.3f}".format(self.step_index, (
                     round(self.stats.stds[-1], 3) if len(self.stats.stds) > 0 else 999), (
                                                                                  round(self.stats.vels[-1].mag(),
                                                                                        3) if len(
                                                                                      self.stats.vels) > 0 else 999)),
                 verticalalignment='center',
                 color='white',
                 bbox=dict(boxstyle="round",
                           ec="darkred",
                           fc="darkred",
                           ),
                 fontsize=10)
        plt.xlim((0, self.w_screen))
        plt.ylim((0, self.h_screen))
        if need_save:
            plt.savefig("sim_t_{}.pdf".format(self.step_index), dpi=1000, format="pdf")
            plt.show(block=True)
        else:
            plt.show(block=False)
            plt.pause(0.04)
        plt.clf()

    def __repr__(self):
        return "Simulator"

    def __str__(self):
        return "<Simulator: step={}>".format(self.step_index)
