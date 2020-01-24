import time
time.sleep(7200)

import os
import sys
import numpy as np
from imgaug import augmenters as iaa
import warnings

MCRNN_DIR = os.path.abspath("../mcrnn")
sys.path.append(MCRNN_DIR)

from files.config import Config
from files import utils
import files.model as modellib
from files import visualize
from files.model import log

MODEL_DIR = os.path.join(MCRNN_DIR, "models")
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    print('Downloading coco')
    utils.download_trained_weights(COCO_MODEL_PATH)

from configs.training_config import CrystalsConfig
from configs.dataset_config import CrystalsDataset

DATASET = sys.argv[1]

print('Training model for  {}'.format(DATASET))
DATASET_DIR = os.path.join("./datasets/", DATASET)

print('Preparing dataset...')
dataset = CrystalsDataset()
dataset.load_crystals(DATASET_DIR, "train")
dataset.prepare()

config = CrystalsConfig()
print('Generating model...')
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
model.load_weights(model.get_imagenet_weights(), by_name=True)

warnings.filterwarnings("ignore")
print('Augmenting images...')
augmentation = iaa.SomeOf((0,2),[iaa.Fliplr(0.5), iaa.Flipud(0.5),
                    iaa.OneOf([iaa.Affine(rotate=90),iaa.Affine(rotate=180),iaa.Affine(rotate=270)]),
                    iaa.Multiply((0.8, 1.5)),iaa.GaussianBlur(sigma=(0.0, 5.0))])

print('Training model... (head layers)')
model.train(dataset, dataset, learning_rate=config.LEARNING_RATE, epochs=30, augmentation=augmentation, layers='heads')
print('Training model... (all layers)')
model.train(dataset, dataset, learning_rate=config.LEARNING_RATE/10, epochs=50, layers="all")
