# library imports
import random
from collections import defaultdict

# project imports
from table import Table


class Qlearner:
    """
    A q-learning model type
    """

    def __init__(self,
                 action_size: int,
                 state_vector_size: int):
        self.state_to_action_dict = defaultdict()
        self.action_size = action_size
        self.state_vector_size = state_vector_size

    def add_sample(self,
                   x: list,
                   y: int):
        try:
            self.state_to_action_dict[tuple(x)][y] += 1
        except:
            self.state_to_action_dict[tuple(x)] = [0 for _ in range(self.action_size)]
            self.state_to_action_dict[tuple(x)][y] += 1

    def predict_max(self,
                    x: list):
        if len(x) < self.state_vector_size:
            x.extend([0 for _ in range(self.state_vector_size - len(x))])
        answer_vector = self.state_to_action_dict[tuple(x)]
        return answer_vector.index(max(answer_vector))

    def predict_explore(self,
                        x: list):
        if len(x) < self.state_vector_size:
            x.extend([0 for _ in range(self.state_vector_size - len(x))])
        answer_vector = self.state_to_action_dict[tuple(x)]
        answer_vector_sum = sum(answer_vector)
        answer_vector = [val/answer_vector_sum for val in answer_vector ]
        return random.choice(list(range(len(answer_vector))), weights=answer_vector)





