# library imports
import os
import math
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class PaperPlots:
    """
    All the plots needed for the paper
    """

    MODEL_TEST_PATH = os.path.join(os.path.dirname(__file__), "model_test_report.json")
    MODEL_NO_COLLISION_TEST_PATH = os.path.join(os.path.dirname(__file__), "model_cm_test_report.json")
    MODEL_RANDOM_TEST_PATH = os.path.join(os.path.dirname(__file__), "model_random_test_report.json")

    def __init__(self):
        pass

    @staticmethod
    def run():
        #PaperPlots.fitness_similarity()
        PaperPlots.models_confusion_matrix([PaperPlots.MODEL_TEST_PATH,
                                            PaperPlots.MODEL_NO_COLLISION_TEST_PATH,
                                            PaperPlots.MODEL_RANDOM_TEST_PATH])

    @staticmethod
    def models_confusion_matrix(test_results_paths: list):
        for test_results_path in test_results_paths:
            with open(test_results_path, "r") as test_data_file:
                data = json.load(test_data_file)
            matrix = data["confusion_matrix"]
            norm_matrix = []
            for row in matrix:
                summer = sum(row)
                norm_matrix.append([item/summer for item in row])
            df = pd.DataFrame(data=norm_matrix)
            sns.heatmap(df, vmin=0, cmap="coolwarm", annot=True, fmt=".5f", linewidths=.5, cbar=False, annot_kws={"fontsize": 6})
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(__file__), "answer", "{}.png".format(os.path.basename(test_results_path))))
            plt.close()

    @staticmethod
    def fitness_similarity():
        answer = {}
        action_index = 0
        answer["keep"] = action_index
        action_index += 1
        for i in range(1, math.ceil(90 / 15) + 1):
            answer["turn_left_{}".format(i * 15)] = action_index
            action_index += 1
        for i in range(1, math.ceil(90 / 15) + 1):
            answer["turn_right_{}".format(i * 15)] = action_index
            action_index += 1
        keys = [val.replace("_", " ").replace("turn", "") for val in list(answer.keys())]
        matrix = [[1.0, 0.9, 0.5, 0.0, 0.0, 0.0, 0.0, 0.9, 0.5, 0.0, 0.0, 0.0, 0.0],
                  [0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.5, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
                  [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
        df = pd.DataFrame(data=matrix, columns=keys, index=keys)
        sns.heatmap(df, vmax=1, vmin=0, cmap="coolwarm", annot=True, fmt=".1f", linewidths=.5, cbar=False)
        plt.tight_layout()
        plt.savefig("fitness_similarity.png")


if __name__ == '__main__':
    PaperPlots.run()
