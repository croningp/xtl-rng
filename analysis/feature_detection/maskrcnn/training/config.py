import os
import sys
import json
import numpy as np
import skimage

# Root directory of the project
ROOT_DIR = os.path.abspath('../../')
MASKRCNN_DIR = os.path.join(ROOT_DIR, 'Mask_RCNN')

# Import Mask RCNN
sys.path.append(MASKRCNN_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.model import log



class CrystalsConfig(Config):
    """Configuration for training on the crystals dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "crystals"
    
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

class CrystalsDataset(utils.Dataset):

    def load_crystals(self, dataset_dir, subset):
        
        # Add classes. We have only one class to add.
        self.add_class("crystal", 1, "polygon")
        #self.add_class("crystal", 2, "circle")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']]
           

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "crystal",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a crystals dataset image, delegate to parent class.
        image_info = self.image_info[image_id]

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]        
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        shapes = info["polygons"]
        count = len(shapes)
        
        for i, p in enumerate(shapes):
            # Get indexes of pixels inside the polygon and set them to 1
            if p['name'] == 'polygon':
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                mask[rr, cc, i] = 1
            if p['name'] == 'circle':
                rr, cc = skimage.draw.circle(p['cy'], p['cx'], p['r'])
                mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # return mask, np.ones([mask.shape[-1]], dtype=np.int32)
    
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s['name']) for s in shapes])
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]

class InferenceConfig(CrystalsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1