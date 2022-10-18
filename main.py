# library imports
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

# project imports
from settings import *
from simulator import Simulator
from hagabub_model import HagabubModel


class Main:
    """

    """

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "hagabub_brain.pickle")
    MODEL_PATH_NO_COLLISION = os.path.join(os.path.dirname(__file__), "hagabub_brain_no_collision.pickle")
    MODEL_PATH_RANDOM = os.path.join(os.path.dirname(__file__), "hagabub_brain_random.pickle")
    MODEL_TEST_PATH = os.path.join(os.path.dirname(__file__), "model_test_report.json")
    MODEL_NO_COLLISION_TEST_PATH = os.path.join(os.path.dirname(__file__), "model_cm_test_report.json")
    MODEL_RANDOM_TEST_PATH = os.path.join(os.path.dirname(__file__), "model_random_test_report.json")
    ANSWER_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "answer")
    TRAIN_SIZE = 10 ** 6 # 10 ** 7
    VALIDATE_SIZE = 10 ** 4 # 10 ** 5
    TEST_SIZE  = 10 ** 5 # 10 ** 5

    def __init__(self):
        pass

    @staticmethod
    def run(need_train: bool = True,
            need_test: bool = True,
            need_simulation: bool = True):

        # IO preparation
        try:
            os.mkdir(Main.ANSWER_FOLDER_PATH)
        except Exception as error:
            pass

        if need_train:
            print("Start training model")
            start_time = time.time()
            hagabub_model = HagabubModel()
            hagabub_model.train(train_size=Main.TRAIN_SIZE,
                                validate_size=Main.VALIDATE_SIZE,
                                see_radius=VISION_RADUIS,
                                max_neighbors=HagabubModel.MAX_N,
                                max_speed=MAX_SPEED)
            print("It took {} seconds to perform the entire train task".format(time.time() - start_time))
            print("Saving the model to: {}".format(Main.MODEL_PATH))
            hagabub_model.save(path=Main.MODEL_PATH)
        else:
            try:
                print("Trying to load the model from: {}".format(Main.MODEL_PATH))
                start_time = time.time()
                hagabub_model = HagabubModel.load(path=Main.MODEL_PATH)
                print("It took {} seconds to load the model".format(time.time() - start_time))
            except:
                print("Was not able to load model, training new model")
                hagabub_model = HagabubModel()
                hagabub_model.train(train_size=Main.TRAIN_SIZE,
                                    validate_size=Main.VALIDATE_SIZE,
                                    see_radius=VISION_RADUIS,
                                    max_neighbors=HagabubModel.MAX_N,
                                    max_speed=MAX_SPEED)
                print("Saving the model to: {}".format(Main.MODEL_PATH))
                hagabub_model.save(path=Main.MODEL_PATH)

        if need_test:
            print("Starting testing the model")
            start_time = time.time()
            test_report = hagabub_model.test(test_size=Main.TEST_SIZE,
                                             see_radius=VISION_RADUIS,
                                             max_neighbors=HagabubModel.MAX_N,
                                             max_speed=MAX_SPEED,
                                             is_max=True)
            print("It took {} seconds to perform the entire test task".format(time.time() - start_time))
            print("Saving testing results to: {}".format(Main.MODEL_TEST_PATH))
            with open(Main.MODEL_TEST_PATH, "w") as test_file:
                json.dump(test_report,
                          test_file,
                          indent=2)

        if need_simulation:
            print("Starting simulator with {} steps".format(MAX_STEPS))
            sim = Simulator(answer_folder=Main.ANSWER_FOLDER_PATH,
                            hagabub_model=hagabub_model,
                            show_visual=True,
                            population_size=POPULATION_SIZE,
                            max_speed=MAX_SPEED,
                            vision_radius=VISION_RADUIS,
                            max_steps=MAX_STEPS,
                            cheat_factor=MAX_STEPS+1)
            sim.run()

    @staticmethod
    def all_train():
        global MODEL_TYPES
        global MODEL_TYPE
        save_paths = [Main.MODEL_PATH, Main.MODEL_PATH_NO_COLLISION, Main.MODEL_PATH_RANDOM]
        test_save_paths = [Main.MODEL_TEST_PATH, Main.MODEL_NO_COLLISION_TEST_PATH, Main.MODEL_RANDOM_TEST_PATH]
        for index in range(len(MODEL_TYPES)):
            MODEL_TYPE = MODEL_TYPES[index]
            print("Start training model: {}".format(MODEL_TYPE))
            start_time = time.time()
            hagabub_model = HagabubModel()
            hagabub_model.train(train_size=Main.TRAIN_SIZE,
                                validate_size=Main.VALIDATE_SIZE,
                                see_radius=VISION_RADUIS,
                                max_neighbors=HagabubModel.MAX_N,
                                max_speed=MAX_SPEED)
            print("It took {} seconds to perform the entire train task".format(time.time() - start_time))
            print("Saving the {} model to: {}".format(MODEL_TYPE, save_paths[index]))
            hagabub_model.save(path=save_paths[index])

            # test models phase
            print("Starting testing the model {}".format(MODEL_TYPE))
            start_time = time.time()
            test_report = hagabub_model.test(test_size=Main.TEST_SIZE,
                                             see_radius=VISION_RADUIS,
                                             max_neighbors=HagabubModel.MAX_N,
                                             max_speed=MAX_SPEED,
                                             is_max=True)
            print("It took {} seconds to perform the entire test task".format(time.time() - start_time))
            print("Saving testing {} results to: {}".format(MODEL_TYPE, test_save_paths[index]))
            with open(test_save_paths[index], "w") as test_file:
                json.dump(test_report,
                          test_file,
                          indent=2)


if __name__ == '__main__':
    #Main.all_train()
    Main.run(need_train=False,
             need_test=False,
             need_simulation=True)
