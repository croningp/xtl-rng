import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
ROOT_PATH = os.path.dirname(HERE_PATH)


MRCNN_PATH = os.path.join(ROOT_PATH, 'Mask_RCNN')

# adding parent directory to path, so we can access the utils easily
import sys
sys.path.append(HERE_PATH)
sys.path.append(MRCNN_PATH)

import logging
logging.basicConfig(level=logging.INFO)

import warnings
warnings.filterwarnings('ignore')

import cv2
import skimage.draw

import numpy as np

import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn import visualize

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon

from detector import Detector

SCAPA_ROOT = '/mnt/scapa4'
ORKNEY_ROOT  ='/mnt/orkney1'

if sys.platform == 'win32':
    SCAPA_ROOT = os.path.join('Z:', os.sep, 'group')
    ORKNEY_ROOT = os.path.join('U:', os.sep)

RAW_XTL_DIR = os.path.join(SCAPA_ROOT, 'Edward Lee', '03-Projects', '07-RandomMOF', 'data', 'Images', 'Crystals')

TARGET_OBJECT = 'MOF5_single_split'    

d = Detector(TARGET_OBJECT)

image_path = os.path.join(RAW_XTL_DIR, 'Initial/MOF5-AC/img_20190705_161424/img_20190705_161424.jpg')

data = d.partition_image(image_path,2,1, 0.3, display='all', save='all', store_data=True)

print(len(data))