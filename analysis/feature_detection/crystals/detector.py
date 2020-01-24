import os
import inspect
import sys
import platform
import skimage

HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

sys.path.append(HERE_PATH)

from files.config import Config
from files import utils
from files import visualize
from files.model import log

if platform.system() == 'Windows':
    import files.model_win as modellib
elif platform.system() == 'Linux':
    import files.model as modellib

from configs import CrystalsConfig
from configs import InferenceConfig


class Detector:

    def __init__(self, experiment):

        self.experiment = experiment
        self.compound = experiment.compound
        self.modeldir = experiment.modeldir
        self.modelpath = experiment.modelpath
        self.crystal_config = CrystalsConfig()
        self.inference_config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode='inference',config=self.inference_config, model_dir = self.modeldir)
        self.model.load_weights(self.modelpath, by_name=True)

    def detect_features(self, image_path):
        image = skimage.io.imread(image_path)
        features = self.model.detect([image], verbose=0)[0]
        return features

    def show_features(self, image_path, features):
        image = skimage.io.imread(image_path)
        visualize.display_instances(image, features['rois'], features['masks'],
                            features['class_ids'], ['BG', 'crystal'])
