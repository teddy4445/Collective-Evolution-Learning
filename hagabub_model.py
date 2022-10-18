# library imports
import math
import numpy
import random
import pickle
from subprocess import call
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# project imports
from table import Table
from vector import Vector
from q_learner import Qlearner
from sklearn.tree import export_graphviz
from settings import VALIDATION_ERROR_DISTANCE, MODEL_TYPE
from model_samples_generator import ModelSamplesGenerator


class HagabubModel:
    """
    The model of the hagabub behavior
    """

    # global #
    PREDICT_INDEX = 0

    # consts #
    TURN_ANGLE = 15
    MAX_N = 5

    # end - consts #

    def __init__(self,
                 q_learning: Qlearner = None,
                 k: int = 9):
        self.action_space = HagabubModel.generate_model_action_space()
        self.sim_action_table = self.action_similarity()
        self._q_learning = q_learning if isinstance(q_learning, Qlearner) else Qlearner(action_size=len(self.action_space),
                                                                                        state_vector_size=self.get_state_size())
        self._k = k
        self._extrapolator = None
        self._extrapolator_params = {}

    def get_extrapolater(self):
        return self._extrapolator, self._extrapolator_params

    @staticmethod
    def generate_model_action_space():
        answer = {}
        action_index = 0
        answer["keep"] = action_index
        action_index += 1
        for i in range(1, math.ceil(90 / HagabubModel.TURN_ANGLE) + 1):
            answer["turn_left_{}".format(i * HagabubModel.TURN_ANGLE)] = action_index
            action_index += 1
        for i in range(1, math.ceil(90 / HagabubModel.TURN_ANGLE) + 1):
            answer["turn_right_{}".format(i * HagabubModel.TURN_ANGLE)] = action_index
            action_index += 1
        return answer

    def action_similarity(self):
        reward_table = Table(columns=list(self.action_space.values()),
                             rows_ids=list(self.action_space.values()))
        reward_table.add_data(new_data=[[1.0, 0.9, 0.5, 0.0, 0.0, 0.0, 0.0, 0.8, 0.5, 0.0, 0.0, 0.0, 0.0],
                                        [0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.5, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
                                        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
                              )
        return reward_table

    def train(self,
              train_size: int,
              validate_size: int,
              max_speed: float,
              see_radius: float,
              max_neighbors: int):
        # generate data to train
        validation_error_distance = 2
        # the data to train the extrapolation
        extrapolator_x = []
        extrapolator_y = []
        while validation_error_distance > VALIDATION_ERROR_DISTANCE:
            sample_generator = ModelSamplesGenerator()
            x, y = sample_generator.generate(samples=train_size,
                                             max_speed=max_speed,
                                             see_radius=see_radius,
                                             max_neighbors=max_neighbors,
                                             is_test=False)
            # learn with q-learning
            for sample_index in range(len(x)):
                self._q_learning.add_sample(x=x[sample_index],
                                            y=y[sample_index])
            for sample_x, sample_y in self._q_learning.state_to_action_dict.items():
                extrapolator_x.append(sample_x)
                extrapolator_y.append(sample_y.index(max(sample_y)))

            # fit random forest
            self._extrapolator = RandomForestClassifier(max_depth=self._k,
                                                        n_estimators=50,
                                                        min_samples_split=6)
            self._extrapolator.fit(extrapolator_x,
                                   extrapolator_y)
            self._extrapolator_params = {"max_depth": self._k,
                                         "min_samples_leaf": 20}

            validation_error_distance = self.evaluate(validate_size=validate_size,
                                                      max_speed=max_speed,
                                                      see_radius=see_radius,
                                                      max_neighbors=max_neighbors)

    def evaluate(self,
                 validate_size: int,
                 max_speed: float,
                 see_radius: float,
                 max_neighbors: int):
        # generate data to test
        sample_generator = ModelSamplesGenerator()
        x, y_true = sample_generator.generate(samples=validate_size,
                                              max_speed=max_speed,
                                              see_radius=see_radius,
                                              max_neighbors=max_neighbors,
                                              is_test=True)
        y_pred = [self.predict_max(neighbors=sample) for sample in x]
        # calc model score with reward function
        rewards = [self.sim_action_table.get(column=y_true[index],
                                             row_id=y_pred[index])
                   for index in range(len(y_pred))]
        # TODO: compute a distribution using (x, rewards) to use in the training phase later
        return numpy.std(rewards)

    def test(self,
             is_max: bool,
             test_size: int,
             max_speed: float,
             see_radius: float,
             max_neighbors: int):
        # generate data to test
        sample_generator = ModelSamplesGenerator()
        x, y_true = sample_generator.generate(samples=test_size,
                                              max_speed=max_speed,
                                              see_radius=see_radius,
                                              max_neighbors=max_neighbors)
        # calc the model's predictions
        if is_max:
            y_pred = [self.predict_max(neighbors=sample) for sample in x]
        else:
            y_pred = [self.predict_explore(neighbors=sample) for sample in x]
        # calc model score with reward function
        rewards = [self.sim_action_table.get(column=y_true[index],
                                             row_id=y_pred[index])
                   for index in range(len(y_pred))]
        # calc error
        return {"test_size": len(y_true),
                "rewards": numpy.mean(rewards),
                "acc": accuracy_score(y_pred=y_pred,
                                      y_true=y_true),
                "top_3_acc": numpy.mean([1 if score >= 0.9 else 0 for score in rewards]),
                "f1": f1_score(y_pred=y_pred,
                               y_true=y_true,
                               average="macro"),
                "confusion_matrix": confusion_matrix(y_true=y_true,
                                                     y_pred=y_pred,
                                                     normalize="all").tolist()
                }

    def predict_explore(self,
                        neighbors: list):
        state = ModelSamplesGenerator.state_neighbors(neighbors=neighbors[2:],
                                                      my_vel=Vector(x=neighbors[0],
                                                                    y=neighbors[1]))
        HagabubModel.PREDICT_INDEX += 1
        if (HagabubModel.PREDICT_INDEX % 1000) == 0:
            print("HagabubModel.PREDICT_INDEX = {}".format(HagabubModel.PREDICT_INDEX))
        try:
            return self._q_learning.predict_explore(x=state)
        except Exception as error:
            return round(self.get_state_size() * random.random())

    def predict_max(self,
                    neighbors: list):
        state = ModelSamplesGenerator.state_neighbors(neighbors=neighbors[2:],
                                                      my_vel=Vector(x=neighbors[0],
                                                                    y=neighbors[1]))
        HagabubModel.PREDICT_INDEX += 1
        if (HagabubModel.PREDICT_INDEX % 1000) == 0:
            print("HagabubModel.PREDICT_INDEX = {}".format(HagabubModel.PREDICT_INDEX))
        try:
            return self._q_learning.predict_max(x=state)
        except Exception as error:
            return list(self._extrapolator.predict(X=[state]))[0]

    def save_extrapolator(self, path: str):
        export_graphviz(self._extrapolator,
                        out_file=path,
                        feature_names=list(self._q_learning.state_to_action_dict.keys()),
                        class_names=list(self.action_space.keys()),
                        rounded=True,
                        proportion=False,
                        precision=2,
                        filled=True)
        call(['dot', '-Tpng', path, '-o', path.replace(".dot", ".png"), '-Gdpi=600'])

    def save(self,
             path: str):
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def load(path: str):
        return pickle.load(open(path, "rb"))

    # help functions #

    @staticmethod
    def get_state_size():
        return 2 + 4 * HagabubModel.MAX_N

    # end - help functions #
