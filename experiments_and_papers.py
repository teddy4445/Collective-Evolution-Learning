# library import
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

# project import
from main import Main
from settings import *
from simulator import Simulator
from hagabub_model import HagabubModel


class ExperimentsWithPlotsForPaper:
    """
    manage all the plots in the project
    """

    REPEAT_COUNT = 5
    DEFAULT_POP_SIZE = 30

    def __init__(self):
        pass

    @staticmethod
    def run_all():
        hagabub_models = ExperimentsWithPlotsForPaper.load_models()
        ExperimentsWithPlotsForPaper.baseline(hagabub_models=hagabub_models)
        ExperimentsWithPlotsForPaper.density_analysis(hagabub_model=hagabub_models["full"])
        ExperimentsWithPlotsForPaper.agent_size_ratio_analysis()
        ExperimentsWithPlotsForPaper.agent_sensing_analysis()
        ExperimentsWithPlotsForPaper.explain_model(hagabub_model=hagabub_models["full"])

    @staticmethod
    def load_models():
        print("Trying to load the model from: {}".format(Main.MODEL_PATH))
        start_time = time.time()
        hagabub_models = {"full": HagabubModel.load(path=Main.MODEL_PATH),
                          "no_collision": HagabubModel.load(path=Main.MODEL_PATH_NO_COLLISION),
                          "random": HagabubModel.load(path=Main.MODEL_PATH_RANDOM)}
        print("It took {} seconds to load the model".format(time.time() - start_time))
        return hagabub_models

    @staticmethod
    def baseline(hagabub_models: dict):
        answer = {}
        for model_name, hagabub_model in hagabub_models.items():
            # calc data
            flocking_signals, collision_signals = ExperimentsWithPlotsForPaper.baseline_flocking_and_collisions(hagabub_model=hagabub_model)
            # calc data
            answer[model_name] = ExperimentsWithPlotsForPaper.baseline_data_prepare(flocking_signals=flocking_signals,
                                                                                    collision_signals=collision_signals)

        # print results #
        # 3 - line plots
        for signal, y_label in [("flocking", "Collective motion index"),
                                ("collision", "Collision rate"),
                                ("objective", "CMCA objective")]:
            for key, name, color, line_shape in [("full", "CMCA model", "green", "-o"),
                                                 ("no_flocking", "CA model", "red", "-^"),
                                                 ("random", "Random model", "black", "-P")]:
                plt.plot(list(range(len(answer["full"]["{}_mean".format(signal)]))),
                         answer[key]["{}_mean".format(signal)],
                         "{}".format(line_shape),
                         color=color,
                         alpha=0.8,
                         label="{}".format(name))
                plt.fill_between(list(range(len(answer["full"]["{}_mean".format(signal)]))),
                                 answer[key]["{}_mean".format(signal)] - answer[key]["{}_std".format(signal)],
                                 answer[key]["{}_mean".format(signal)] + answer[key]["{}_std".format(signal)],
                                 alpha=0.1,
                                 color=color)
            plt.xlabel("Simulation step")
            plt.ylabel("{}".format(y_label))
            plt.grid(alpha=0.2,
                     color="gray")
            plt.legend()
            plt.savefig(os.path.join(Main.ANSWER_FOLDER_PATH, "{}_experiment_baseline.png".format(signal)))
            plt.close()
        # bar plot with the summary
        bar_index = 0
        width = 0.3
        for signal, color, y_label in [("flocking", "blue", "Collective motion index"),
                                       ("collision", "red",  "Collision rate"),
                                       ("objective", "green",  "CMCA objective")]:
            x = [val - 1 * width + bar_index * width for val in range(len(answer))]
            y = [np.mean(answer[key]["{}_mean".format(signal)]) for key, item in answer.items()]
            plt.bar(x,
                    y,
                    width=0.3,
                    color=color,
                    label="{}".format(y_label))
            y_err = [np.mean(answer[key]["{}_std".format(signal)]) for key, item in answer.items()]
            plt.errorbar(x, y, yerr=y_err, fmt="o", color="b", capsize=2)
            bar_index += 1
        plt.xticks(range(len(answer)), [key for key in answer.keys()])
        plt.ylabel("Model's score")
        plt.legend()
        plt.savefig(os.path.join(Main.ANSWER_FOLDER_PATH, "experiment_bar_summary_baseline.png"))
        plt.close()

    @staticmethod
    def density_analysis(hagabub_model):
        x = []
        y = []
        y_err = []
        x_values = list(range(10, 160, 10))
        for pop_size in x_values:
            # calc data
            flocking_signals, collision_signals = ExperimentsWithPlotsForPaper.baseline_flocking_and_collisions(hagabub_model=hagabub_model,
                                                                                                                pop_size=pop_size)
            data_answer = ExperimentsWithPlotsForPaper.baseline_data_prepare(flocking_signals=flocking_signals,
                                                                             collision_signals=collision_signals)
            x.append(pop_size)
            y.append(np.mean(data_answer["objective_mean"]))
            y_err.append(np.mean(data_answer["objective_std"]))
        # save plot
        plt.errorbar(x,
                     y,
                     y_err,
                     ecolor="black",
                     color="blue",
                     capsize=3)
        plt.xlabel("Population Size [1]")
        plt.ylabel("Model's score [1]")
        plt.ylim((0, max([y[i] + y_err[i] for i in range(len(x))])))
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(__file__), "answer", "summary_ratio_sensitivity_analysis.png"))
        plt.close()

    @staticmethod
    def agent_size_ratio_analysis():
        global HAGABUB_WIDTH
        global HAGABUB_LENGTH
        x = []
        y = []
        y_err = []
        ratios = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for ratio in ratios:
            # train model
            HAGABUB_LENGTH = HAGABUB_WIDTH * ratio
            hagabub_model = HagabubModel()
            hagabub_model.train(train_size=Main.TRAIN_SIZE,
                                validate_size=Main.VALIDATE_SIZE,
                                see_radius=VISION_RADUIS,
                                max_neighbors=HagabubModel.MAX_N,
                                max_speed=MAX_SPEED)
            # calc data
            flocking_signals, collision_signals = ExperimentsWithPlotsForPaper.baseline_flocking_and_collisions(hagabub_model=hagabub_model)
            data_answer = ExperimentsWithPlotsForPaper.baseline_data_prepare(flocking_signals=flocking_signals,
                                                                             collision_signals=collision_signals)
            x.append(ratio)
            y.append(np.mean(data_answer["objective_mean"]))
            y_err.append(np.mean(data_answer["objective_std"]))
        # save plot
        plt.errorbar(x,
                     y,
                     y_err,
                     ecolor="black",
                     color="blue",
                     capsize=3)
        plt.xlabel("Agent's length to width ratio [1]")
        plt.ylabel("Model's score [1]")
        plt.ylim((0, max([y[i] + y_err[i] for i in range(len(x))])))
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(__file__), "answer", "summary_ratio_sensitivity_analysis.png"))
        plt.close()

    @staticmethod
    def agent_sensing_analysis():
        global HAGABUB_LENGTH
        x = []
        y = []
        y_err = []
        radiuses = [HAGABUB_LENGTH*(i+1) for i in range(6)]
        for radius in radiuses:
            # train model
            hagabub_model = HagabubModel()
            hagabub_model.train(train_size=Main.TRAIN_SIZE,
                                validate_size=Main.VALIDATE_SIZE,
                                see_radius=radius,
                                max_neighbors=HagabubModel.MAX_N,
                                max_speed=MAX_SPEED)
            # calc data
            flocking_signals, collision_signals = ExperimentsWithPlotsForPaper.baseline_flocking_and_collisions(hagabub_model=hagabub_model)
            data_answer = ExperimentsWithPlotsForPaper.baseline_data_prepare(flocking_signals=flocking_signals,
                                                                             collision_signals=collision_signals)
            x.append(radius)
            y.append(np.mean(data_answer["objective_mean"]))
            y_err.append(np.mean(data_answer["objective_std"]))
        # save plot
        plt.errorbar(x,
                     y,
                     y_err,
                     ecolor="black",
                     color="blue",
                     capsize=3)
        plt.xlabel("Agent's Sensing radius [1]")
        plt.ylabel("Model's score [1]")
        plt.ylim((0, max([y[i] + y_err[i] for i in range(len(x))])))
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(__file__), "answer", "summary_sensing_sensitivity_analysis.png"))
        plt.close()

    @staticmethod
    def explain_model(hagabub_model):
        hagabub_model.save_extrapolator(path=os.path.join(Main.ANSWER_FOLDER_PATH, "explain_tree_plot.dot"))

    @staticmethod
    def baseline_flocking_and_collisions(hagabub_model,
                                         pop_size: int = POPULATION_SIZE,
                                         save_baseline_plots: bool = False):
        if pop_size == 0:
            pop_size = ExperimentsWithPlotsForPaper.DEFAULT_POP_SIZE

        flocking_signals = []
        collision_signals = []
        for _ in range(ExperimentsWithPlotsForPaper.REPEAT_COUNT):
            sim = Simulator(answer_folder=Main.ANSWER_FOLDER_PATH,
                            hagabub_model=hagabub_model,
                            show_visual=False,
                            population_size=pop_size,
                            max_speed=MAX_SPEED,
                            vision_radius=VISION_RADUIS,
                            max_steps=MAX_STEPS,
                            cheat_factor=MAX_STEPS + 1)
            sim.run()
            flocking_signals.append(sim.stats.stds)
            collision_signals.append(sim.stats.hits)

        # check if we want to save the process graphs
        if save_baseline_plots:
            # print stuff #
            # circular STD
            for index in range(ExperimentsWithPlotsForPaper.REPEAT_COUNT):
                plt.plot(list(range(MAX_STEPS)),
                         flocking_signals[index],
                         "-o",
                         color="black",
                         alpha=0.1,
                         markersize=2,
                         linewidth=1)
            plt.plot(list(range(MAX_STEPS)),
                     np.mean(flocking_signals, axis=0),
                     "-",
                     color="blue",
                     alpha=1,
                     linewidth=2)
            plt.ylabel("Circular STD [1]")
            plt.xlabel("Simulation step [1]")
            plt.savefig(os.path.join(os.path.dirname(__file__), "answer", "baseline_circular_std_{}.png".format(pop_size)))
            plt.ylim((0, 3))
            plt.savefig(os.path.join(os.path.dirname(__file__), "answer", "baseline_circular_std_full_y_{}.png".format(pop_size)))
            plt.close()

            # collision rate
            for index in range(ExperimentsWithPlotsForPaper.REPEAT_COUNT):
                plt.plot(list(range(MAX_STEPS)),
                         collision_signals[index],
                         "-o",
                         color="black",
                         alpha=0.1,
                         markersize=2,
                         linewidth=1)
            plt.plot(list(range(MAX_STEPS)),
                     np.mean(collision_signals, axis=0),
                     "-",
                     color="red",
                     alpha=1,
                     linewidth=2)
            plt.ylabel("Collision rate [1]")
            plt.xlabel("Simulation step [1]")
            plt.savefig(os.path.join(os.path.dirname(__file__), "answer", "baseline_collision_rate_{}.png".format(pop_size)))
            plt.ylim((0, 1))
            plt.savefig(os.path.join(os.path.dirname(__file__), "answer", "baseline_collision_rate_full_y_{}.png".format(pop_size)))
            plt.close()
        # return the results
        return flocking_signals, collision_signals

    @staticmethod
    def baseline_data_prepare(flocking_signals,
                              collision_signals):
        # calc signals
        flocking_signal = np.mean(flocking_signals, axis=0)
        flocking_signal_std = np.mean(flocking_signals, axis=0)
        collision_signal = np.mean(collision_signals, axis=0)
        collision_signal_std = np.mean(flocking_signals, axis=0)
        final_signal = np.asarray([flocking_signal[index] * 0.5 + (1 - collision_signal[index]) * 0.5
                                   for index in range(len(flocking_signal))])
        final_signal_std = np.asarray([flocking_signal_std[index] * 0.5 + collision_signal_std[index] * 0.5
                                       for index in range(len(flocking_signal_std))])
        # save in the answer
        return {
            "flocking_mean": flocking_signal,
            "flocking_std": flocking_signal_std,
            "collision_mean": collision_signal,
            "collision_std": collision_signal_std,
            "objective_mean": final_signal,
            "objective_std": final_signal_std
        }


if __name__ == '__main__':
    ExperimentsWithPlotsForPaper.run_all()
