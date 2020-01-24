import os
import csv
import cv2
import sys
import inspect
import logging
import skimage.draw
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
# %matplotlib inline

# add datetime to every logging message
logging.basicConfig(level=logging.INFO)
logging.basicConfig( format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

# this get our current location in the file system
HERE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
MRCNN_PATH = os.path.join(HERE_PATH, 'Mask_RCNN')
# adding parent directory to path, so we can access the utils easily
sys.path.append(HERE_PATH)
sys.path.append(MRCNN_PATH)

# import mrcnn libraries
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn import visualize
from mrcnn.config import Config

from __main__ import TARGET_OBJECT


# determine if we are using Windows or Linux filing conventions and assign root folders
if sys.platform == 'win32':
    SCAPA_ROOT = os.path.join('Z:', os.sep, 'group')
    ORKNEY_ROOT = os.path.join('U:', os.sep)
else:
    SCAPA_ROOT = '/mnt/scapa4'
    ORKNEY_ROOT  ='/mnt/orkney1'

ORKNEY_TEAM = os.path.join(ORKNEY_ROOT, 'Clusters')
ORKNEY_PROJECT = os.path.join(ORKNEY_TEAM, 'RandomXtl')
ORKNEY_MASKRCNN = os.path.join(ORKNEY_PROJECT, 'MaskRCNN')
ORKNEY_DETECTED = os.path.join(ORKNEY_MASKRCNN, 'detected')
ORKNEY_TRAINING = os.path.join(ORKNEY_MASKRCNN, 'training')
ORKNEY_DATASETS = os.path.join(ORKNEY_TRAINING, 'datasets')
ORKNEY_LOGS = os.path.join(ORKNEY_TRAINING, 'logs')

ORKNEY_IMAGES = os.path.join(ORKNEY_PROJECT, 'images')
ORKNEY_RAW_IMGS = os.path.join(ORKNEY_IMAGES, 'raw_images')
ORKNEY_PART_IMGS = os.path.join(ORKNEY_IMAGES, 'partitioned')

ORKNEY_CV = os.path.join(ORKNEY_PROJECT, 'computer_vision')

IMG_EXTS = ['.img', '.bmp', '.tiff', '.jpg', '.png']



def get_models():
    models = {}
    for model_dir in os.listdir(ORKNEY_LOGS):
        model_dir_path = os.path.join(ORKNEY_LOGS, model_dir)
        object_name, timestamp = model_dir[:-13], model_dir[-13:]
        epoch_names = sorted(filter(lambda f: f.endswith('.h5'), os.listdir(model_dir_path)))
        epoch_paths = [os.path.join(model_dir_path, f) for f in epoch_names]
        epochs = len(epoch_paths)

        if object_name not in models.keys():
            models[object_name] = []
        this_dict = {'name': model_dir, 'stamp': timestamp, 'path': model_dir_path,
                     'epochs': epochs, 'epoch_paths': epoch_paths}
        models[object_name].append(this_dict)
    return models
    
def show_models():
    
    print('Object Name'.ljust(21)+'Timestamp'.ljust(14) +'Epochs')
    for k, v in MODELS.items():   
        for data in v:
            print(k.ljust(20), data['stamp'].ljust(17), data['epochs'])    
            
MODELS = get_models()
show_models()

class Detector:

    def __init__(self, target_object, model_version=-1, epoch=-1):
             
        self.target_object = target_object.lower()
        self.output_root_dir = os.path.join(ORKNEY_DETECTED, target_object.lower())
        
        self.inference_config = InferenceConfig()
        self.crystals_config = CrystalsConfig()

        self.model = modellib.MaskRCNN(mode="inference", 
                          config=self.inference_config,
                          model_dir=ORKNEY_LOGS)
        self.model_path = MODELS[self.target_object][model_version]['epoch_paths'][epoch]
        self.model.load_weights(self.model_path, by_name=True)
        

            
            
    def detect_path(self, input_path, partitioning = [2,2,0.1], save=False, show=False):
        # unpack arguments
        self.input_path = input_path
        self.rows, self.cols, self.overlap = partitioning
        self.save = save
        self.show = show
        
        # assign folders for output files
        self.image_title, self.image_ext = os.path.basename(input_path).split('.')

        self.reaction_path = os.path.dirname(os.path.dirname(input_path))
        self.rxn_id = os.path.basename(self.reaction_path)
        self.exp_path = os.path.dirname(self.reaction_path)
        self.exp_id = os.path.basename(self.exp_path)
        
        self.output_dir = os.path.join(self.output_root_dir, self.exp_id, self.rxn_id, self.image_title)
        os.makedirs(self.output_dir, exist_ok=True)

        # duplicate image to output folder
        self.duplicate_path = os.path.join(self.output_dir, os.path.basename(input_path))
        cv2.imwrite(self.duplicate_path, cv2.imread(input_path))
        
        #load and reverse image (maskrccn does weird things pparaently....)
        self.img_bgr = cv2.imread(input_path)       
        self.img = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)

        # self.image_rev = self.img[:,::-1]
        self.image_height, self.image_width = self.img.shape[:2]  
            
        self.get_rois(self.img, self.rows, self.cols, self.overlap)

       # dict for collating detections from separate image chunks in the same format as maskrcnn
        self.all_rs = {'rois':np.empty((0,4)),
                      'masks':np.empty((self.image_height, self.image_width, 0)),
                      'class_ids':np.empty(0)}
        
        self.detection_data = []
        
        for idx, roi in enumerate(self.roi_regions):

            image_roi = self.img[roi['t']:roi['b'], roi['l']: roi['r']]
            image_roi_rev = image_roi[:,::-1]

            logging.info('Detecting from chunk: {}/{}, {}:{}, {}:{}]'.format(\
                    idx+1, len(self.roi_regions), roi['t'],roi['b'],roi['t'], roi['t']))    
            
            roi_detections = self.model.detect([image_roi_rev], verbose=1)[0]            
            ax = self.display_instances(image_roi_rev, roi_detections['rois'], roi_detections['masks'], roi_detections['class_ids'], 
                                     ['BG', 'polygon'])
            self.save_roi_detections(roi, ax)
            self.collate_detections(roi, roi_detections)

        ax = self.display_instances(self.img, self.all_rs['rois'], self.all_rs['masks'], 
                               self.all_rs['class_ids'], ['BG', 'polygon'])   
        self.save_image_detections(ax)
        self.save_data()
    plt.clf()
    
    def save_data(self):
        # name contains 'image title';'chunking settings', 'overlapping settings'
        new_csv_name ='{};[{},{},{}].csv'.format(self.image_title,self.rows, self.cols, self.overlap)
        # save folder contains 'chunking settings', 'overlapping settings'
        save_folder = os.path.join(self.output_dir, '[{},{},{}]'.format(self.rows, self.cols, self.overlap), 'data')
        
        csv_file = os.path.join(save_folder, new_csv_name) 
        logging.info('Saving compressed data (csv and masks) to {}'.format(save_folder))
        os.makedirs(save_folder, exist_ok=True)
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['path','top', 'left', 'id'])
            for idx, detection in enumerate(self.detection_data):
                top, left = detection['topleft']
                path = os.path.join(save_folder, '{}.png'.format(str(idx).zfill(5)))
                writer.writerow([path, top, left, detection['id']])
                cv2.imwrite(path, detection['outline'])

    def save_roi_detections(self, roi, ax):

        new_image_name ='{};({}-{})({}-{}).png'.format(self.image_title,roi['t'],roi['b'],roi['l'],roi['r'],self.image_ext)
        # save folder contains 'chunking settings', 'overlapping settings'
        save_folder = os.path.join(self.output_dir, '[{},{},{}]'.format(self.rows, self.cols, self.overlap), 'images')                  
        save_path = os.path.join(save_folder, new_image_name)
        logging.info('Saving image chunk to {}'.format(save_path))
        os.makedirs(save_folder, exist_ok=True)

        plt.savefig(save_path)  

    def save_image_detections(self, ax):
        # name contains 'image title';'chunking settings', 'overlapping settings'
        new_image_name ='{};[{},{},{}].png'.format(self.image_title,self.rows, self.cols, self.overlap)
        # save folder contains 'chunking settings', 'overlapping settings'
        save_folder = os.path.join(self.output_dir, '[{},{},{}]'.format(self.rows, self.cols, self.overlap))                 
        save_path = os.path.join(save_folder, new_image_name)
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(save_path)   
            
    def get_rois(self, image, rows=6, cols=6, overlap=0.1):
        
        self.rows = rows
        self.cols = cols
        self.overlap = overlap
        self.image_height, self.image_width = image.shape[:2]  
        self.roi_width = int(self.image_width/self.cols)
        self.roi_height = int(self.image_height/self.rows)
        self.overlap_width = int(self.roi_width*self.overlap)
        self.overlap_height = int(self.roi_height*self.overlap)        
        self.roi_regions = []
        
        for n in range(rows):
            #first divide image into n equal rows     
            #roi rows without overlap
            top1 = int(n * self.roi_height)
            bottom1 = int(top1+self.roi_height)
            #extend roi rows by overlap
            top2 = max(0, top1-self.overlap_height)
            bottom2 = min(self.image_height, bottom1+self.overlap_height)      
            
            for m in range(cols):
                # then divide rows vertically m times to get roi chunk
                #roi columns without overlap 
                left1 = int(m * self.roi_width)
                right1 = int(left1+self.roi_width)
                # roi columns with overlap
                left2 = max(0, left1-self.overlap_width)
                right2 = min(self.image_width, right1+self.overlap_width)
                roi_data = {'t':top2, 'b':bottom2, 'l':left2, 'r':right2}
                self.roi_regions.append(roi_data)
               
    def collate_detections(self, roi, detections):
        collated_data = []
        masks_in_chunk = detections['masks'].T
        print(masks_in_chunk.shape)
        for idx in range(len(detections['rois'])):
            self.all_rs['class_ids'] = np.append(self.all_rs['class_ids'], detections['class_ids'][idx]).astype(int)    
            ##### Deal with rois #####
            # specify roi of current detection (top, left, bottom, right)

            _t,_l,_b,_r = roi_in_chunk = detections['rois'][idx]

            # get roi location in original image
            t, b = [rownum + roi['t'] for rownum in [_t, _b]]
            l, r = [colnum + roi['l'] for colnum in [_l, _r]]
            print(roi)
            print(_t, t)
            print(_b, b)
            print(_l, l)
            print(_r, r)
            roi_in_origin = np.array([t,l,b,r])

            # append roi to dict for original image
            self.all_rs['rois'] = np.vstack([self.all_rs['rois'], roi_in_origin])   

            ##### Deal with masks #####
            # specify mask of current detection
            mask_in_chunk = masks_in_chunk[idx].T

            print(mask_in_chunk.shape, roi_in_origin)

            # create a blank canvas to represent original image
            mask_in_origin = np.zeros(self.img.shape[:2], np.uint8)
            # put chunk into original position of canvas
            mask_in_origin[roi['t']:roi['b'], roi['l']:roi['r']] = mask_in_chunk
            # add mask to collated dict
            self.all_rs['masks'] = np.dstack([self.all_rs['masks'], mask_in_origin])

            # collate detection data in form that can be compressed easily
            # note that maskrcnn flips the image horizontally, so this data points are also reversed
            mask = mask_in_origin[t:b,l:r][:,::-1]
            mask[mask == True] = 255

            chunk_data = {'topleft':(t,self.image_width-r), 
                              'outline':mask, 
                              'id':detections['class_ids'][idx]}

            self.detection_data.append(chunk_data)   

    def display_instances(self, image, boxes, masks, class_ids, class_names,
                        scores=None, title="",
                        figsize=(16, 16), ax=None,
                        show_mask=True, show_bbox=True,
                        colors=None, captions=None):
        # Number of instances
        N = boxes.shape[0]

        if not N:
            logging.info("\n*** No instances to display *** \n")
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

        _, ax = plt.subplots(1, figsize=figsize)

        # Generate random colors
        colors = colors or visualize.random_colors(N)

        # Show area outside image boundaries.
        height, width = image.shape[:2]
        ax.set_ylim(height)
        ax.set_xlim(width)
        ax.axis('off')

        masked_image = image.astype(np.uint32).copy()
 
        for i in range(N):
            color = colors[i]

            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            if show_bbox:
                p = visualize.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                    alpha=0.7, linestyle="dashed",
                                    edgecolor=color, facecolor='none')
                ax.add_patch(p)

            # Label
            if not captions:
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]
            ax.text(x1, y1 + 8, caption,
                    color='w', size=11, backgroundcolor="none")

            # Mask
            mask = masks[:, :, i]
            if show_mask:
                masked_image = visualize.apply_mask(masked_image, mask, color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = visualize.find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
        ax.imshow(masked_image.astype(np.uint8))
        return ax


# TARGET_OBJECT = 'W19'   
# model_version = 0
# d = Detector(TARGET_OBJECT)

# input_path = '/mnt/orkney1/Chemobot/crystalbot_imgs/W19/20180214-0/Reaction_030/Images/Image_016.png'
# d.detect_path(input_path)

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
    