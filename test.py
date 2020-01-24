import os
import sys
import inspect


import cv2

from robot.tools import robot
from robot.tools import experiment

HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

sys.path.insert(0, HERE_PATH)


def load_exp(exp_name):
    new_experiment = experiment.Experiment(exp_name)
    return new_experiment

def load_plat(args=None):
    plat = robot.Robot()
    if args:
        if 'a' in args:
            plat.load_arduino()
        if 'p' in args:
            plat.load_pumps()
        if 'c' in args:
            plat.load_camera()
    return plat
