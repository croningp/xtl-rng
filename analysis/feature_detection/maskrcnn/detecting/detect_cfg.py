import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

MRCNN_PATH = os.path.join(HERE_PATH, 'Mask_RCNN')

# adding parent directory to path, so we can access the utils easily
import sys
sys.path.append(HERE_PATH)
sys.path.append(MRCNN_PATH)

import logging
logging.basicConfig(level=logging.INFO)

from mrcnn.config import Config

from __main__ import TARGET_OBJECT

class CrystalsConfig(Config):


    """Configuration for training on the crystals dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """

    NAME = TARGET_OBJECT
    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    # From Nucleus example
    BACKBONE = "resnet101"

    # This computer has a GTX1060
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shape for crystals
    
    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    # From Nucleus example
    DETECTION_MIN_CONFIDENCE = 0
    
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    # From Nucleus example
    RPN_NMS_THRESHOLD = 0.9
    
    # Maximum number of ground truth instances to use in one image
    # From Nucleus example
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    # From Nucleus example
    DETECTION_MAX_INSTANCES = 200

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = int(800/2)
    IMAGE_MAX_DIM = int(1280/2)
    # Input image resizing
    # Random crops of size 512x512
    #IMAGE_RESIZE_MODE = "crop"
    #IMAGE_MIN_DIM = 512
    #IMAGE_MAX_DIM = 512
    #IMAGE_MIN_SCALE = 2.0

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)#, 512)

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 50

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 300

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

 
class InferenceConfig(CrystalsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    