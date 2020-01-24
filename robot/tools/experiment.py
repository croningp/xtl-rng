import os
import sys
import json
import time
import shutil


from robot.tools.robot import Robot
from robot.tools.reaction import Reaction
from robot.tools.image import Image

from robot.utils.filing import FilingSystem

from analysis.rng import Generator

PLATFORM_ROOT = sys.path[0]
INPUT_DATA_ROOT = os.path.join(PLATFORM_ROOT, 'experiments')


class Experiment:

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name  # name of experiment
        self.compound = experiment_name         # name of compound being formed

        self.remote_root = os.path.join(PLATFORM_ROOT, 'results')   # remote root directory for saving data
        self.local_root = os.path.join(PLATFORM_ROOT, 'results')    # local data root (incase of loss of connection)

        # paths for image analysis
        self.mcrnndir = os.path.join(PLATFORM_ROOT, 'analysis', 'feature_detection', 'crystals')
        self.modeldir = os.path.join(self.mcrnndir, 'models')
        self.modelpath = os.path.join(self.modeldir, '{}.h5'.format(self.compound))


        self.load_data()                        # gets parameters for the reaction components
        self.load_robot()                       # gets parameters for the robotic and activates robot
        self.setup_filing()                     # defines all save folders, files, etc
        self.load_processes()                   # loads necessary software

    def start(self):
        # two components in an experiment. Reactions are all started, then images taken.
        self.run_reactions()
        self.run_images()
        sys.exit()

    def load_data(self):
        # json contains data specifying reagent volumes, pumping times, etc
        self.input_folder = os.path.join(INPUT_DATA_ROOT, self.experiment_name)
        self.configfile = os.path.join(self.input_folder, 'config.json')

        self.data = json.load(open(self.configfile))

        self.number_of_reactions = self.data['number_of_reactions']
        self.images_per_reaction = self.data['images_per_reaction']
        self.time_between_images = self.data['time_between_images']
        self.reagent_data = self.data['reagents']

    def setup_filing(self):
        #gets all files ready at the start
        self.filing = FilingSystem(self)
        os.makedirs(self.filing.local_exp_folder, exist_ok=True)
        os.makedirs(self.filing.remote_exp_folder, exist_ok=True)
        
        # creates a duplicate of the input reaction file
        self.configfile_copypath1 = os.path.join(self.filing.local_exp_folder, 'config.json')
        self.configfile_copypath2 = os.path.join(self.filing.remote_exp_folder, 'config.json')
        shutil.copyfile(self.configfile, self.configfile_copypath1)
        shutil.copyfile(self.configfile, self.configfile_copypath2)

    def load_robot(self):
        # instantiated and prepares pumps, camera and motors which can be accessed using self.robot
        self.robot = Robot()
        self.robot.load_hardware()

    def load_processes(self):

        self.reaction = Reaction(self)  # contains algorithm of moving to correct location and pumping reagents based on reaction number
        self.image = Image(self)        # contains algorithm of moving to correct location and taking/saving an image based on reaction number and image number
        self.xtlrng = Generator(self)   # contains algorithm for detecting features in images and converting this to a random number

    def run_reactions(self):
        # reactions are indexed and run in order
        for index in range(self.number_of_reactions):
            self.reaction.run(index)
        # robot is homed
        self.robot.BOT.home()

    def run_images(self):
        # determine when to start taking each set of image
        self.schedule = [time.time()+i*self.time_between_images for i in range(self.images_per_reaction)]
        for image in range(self.images_per_reaction):
            while time.time() < self.schedule[image]:
                time.sleep(1)
            # take one image of each reaction at this time to create a set
            for reaction in range(self.number_of_reactions):
                self.image.run(reaction, image)
                self.xtlrng.run(reaction, image)
            # robot is homed
            self.robot.BOT.home()
