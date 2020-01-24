import os
import sys
import inspect

from robot.tools import experiment

HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

sys.path.insert(0, HERE_PATH)

if __name__ == '__main__':
    """ Experiment name is the name of the compound being formed.
        There must be a config file with this name in /experiments/configs which contains the reaction data
        It is also used as the root directory in the results directory for data storage.
    """
    experiment_name = sys.argv[1]
    # create new experiment with experiment name
    new_experiment = experiment.Experiment(experiment_name)
    # begin experiment
    new_experiment.start()
