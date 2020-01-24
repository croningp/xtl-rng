import os
import csv
import cv2
import sys
import inspect
import logging
from shutil import copyfile
import skimage.draw
import skimage.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon

# add datetime to every logging message
logging.basicConfig(level=logging.INFO)
logging.basicConfig( format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

# this get our current location in the file system
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
MRCNN_PATH = os.path.join(HERE_PATH, 'Mask_RCNN')
# adding parent directory to path, so we can access the utils easily
sys.path.append(HERE_PATH)
sys.path.append(MRCNN_PATH)

#import mrcnn libraries
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn import visualize
from mrcnn.config import Config


screen_dpi = 72


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

def lin2win(filepath):
    return filepath.replace('/mnt/scapa4', 'Z:').replace('/mnt/orkney1', 'U:').replace('/', '\\')

def win2lin(filepath):
    return filepath.replace('Z:', '/mnt/scapa4' ).replace('U:', '/mnt/orkney1').replace('\\', '/')

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

TARGET_OBJECT = 'w19'
model_version = -1
epoch = -1
model_path = MODELS[TARGET_OBJECT][model_version]['epoch_paths'][epoch]
output_root_dir = os.path.join(HERE_PATH, TARGET_OBJECT.lower())
exp_path = '/mnt/orkney1/Chemobot/crystalbot_imgs/Co4/20180216-0'
partitioning=[6,6,0.2]
reaction_range=[0,-1]
image_range=[70,71]
overwrite=True
class CrystalsConfig(Config):

    NAME = 'crystal'
    BACKBONE = "resnet101"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1 
    DETECTION_MIN_CONFIDENCE = 0
    RPN_NMS_THRESHOLD = 0.9
    MAX_GT_INSTANCES = 200
    DETECTION_MAX_INSTANCES = 200
    IMAGE_MIN_DIM = int(800/2)
    IMAGE_MAX_DIM = int(1280/2)
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 50
    STEPS_PER_EPOCH = 300
    VALIDATION_STEPS = 5

 
class InferenceConfig(CrystalsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

crystal_config = CrystalsConfig()
inference_config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=ORKNEY_LOGS)
model.load_weights(model_path, by_name=True)

def get_partitions(image, partitioning=[1,1,0]):

    rows, cols, overlap = partitioning
    image_height, image_width = image.shape[:2] 
    
    roi_width = int(image_width/cols)
    roi_height = int(image_height/rows)
    overlap_width = int(roi_width*overlap)
    overlap_height = int(roi_height*overlap) 
    
    roi_regions = []

    for n in range(rows):
        #first divide image into n equal rows     
        #roi rows without overlap
        top1 = int(n * roi_height)
        bottom1 = int(top1+roi_height)
        #extend roi rows by overlap
        top2 = max(0, top1-overlap_height)
        bottom2 = min(image_height, bottom1+overlap_height)      

        for m in range(cols):
            # then divide rows vertically m times to get roi chunk
            #roi columns without overlap 
            left1 = int(m * roi_width)
            right1 = int(left1+roi_width)
            # roi columns with overlap
            left2 = max(0, left1-overlap_width)
            right2 = min(image_width, right1+overlap_width)
            roi_data = {'t':top2, 'b':bottom2, 'l':left2, 'r':right2}
            roi_regions.append(roi_data)
    return roi_regions

def detect_path(input_path, partitioning=[1,1,0], overwrite=False):
    
    # get data to do with image and where to save it......
    rows, cols, overlap = partitioning
    image_head, image_ext = os.path.splitext(input_path)
    image_title = os.path.basename(image_head)
    
    reaction_path = os.path.dirname(os.path.dirname(input_path))
    rxn_id = os.path.basename(reaction_path)
    exp_path = os.path.dirname(reaction_path)
    exp_id = os.path.basename(exp_path)
    
    output_dir = os.path.join(output_root_dir, exp_id, rxn_id, image_title)
    os.makedirs(output_dir, exist_ok=True)
    
    image = skimage.io.imread(input_path)
    
    duplicate_path = os.path.join(output_dir, os.path.basename(input_path))
    skimage.io.imsave(duplicate_path, image)
    
    image_height, image_width = image.shape[:2]  

    # partition the image as described by rows, cols and overlap
    partitions = get_partitions(image, partitioning)  # a list of dicts with keys: 't', 'b', 'l', 'r'
    # array with the same dimensions as partitioned
    partition_shape = np.reshape(range(rows*cols), (rows, cols))
    
    # data to be collected
    all_rois = np.empty((0, 4))
    all_masks = np.empty((image_height, image_width, 0))
    all_ids = np.empty(0)
    masks_per_partition = []    
    
    fig, ax = plt.subplots()
    for part_idx, p in enumerate(partitions):
        
        # data about partition and neighbours. Need to check each neighbour for overlapping detections
        partition_coordinates = np.where(partition_shape == part_idx)
        x, y = partition_coordinates[0][0], partition_coordinates[1][0]

        prev_neighbour_coordinates = [[x-1, y], [x,y-1]]
        neighbours_to_check = [i for i in prev_neighbour_coordinates if  -1 not in i]
        indexes_to_check = [partition_shape[x][y] for x, y in neighbours_to_check]
        
        # actual image to be detected
        image_partition = image[p['t']:p['b'], p['l']:p['r']]
        
        save_root = os.path.join(output_dir, '[{},{},{}]'.format(rows, cols, overlap))
        save_folder = os.path.join(save_root, 'images')
        os.makedirs(save_folder, exist_ok=True)
        new_image_name ='{};({}-{})({}-{}).png'.format(image_title,p['t'],p['b'],p['l'],p['r'])
        save_path = os.path.join(save_folder, new_image_name)
        if not overwrite: # if we don't want to overwrite           
            if os.path.exists(save_path):  # if the path already exists
                continue  # skip and go to next partition

        # detections
        p_detections = model.detect([image_partition], verbose = 1)[0]       
        visualize.display_instances(image_partition, p_detections['rois'], p_detections['masks'],
                                        p_detections['class_ids'], ['BG', ''], ax = ax)

        # saving data
        plt.savefig(save_path) 
        print('saving labelled image', save_path)
        
        # reset matplotlib
        plt.clf()
        plt.close('all')
        fig, ax = plt.subplots(figsize = (int(image_height/screen_dpi), int(image_width/screen_dpi)))
        
        # can only select mask by index when the mask array is transposed
        masks = p_detections['masks'].T              
        masks_in_partition = []
        # iterate through each detected mask
        for roi_idx, roi in enumerate(p_detections['rois']):
            _t, _l, _b, _r = roi        # roi in partition
            
            # corresponding roi in original image
            t, b = [rownum + p['t'] for rownum in [_t, _b]]
            l, r = [colnum + p['l'] for colnum in [_l, _r]]

            # add roi to roi data
            all_rois = np.vstack([all_rois, np.array([t,l,b,r])])     
            
            # get array of just the mask. Transposition needed because of format.
            mask = masks[roi_idx].T[_t:_b,_l:_r]   
            
            # put mask array into original image dimensions
            mask_in_origin = np.zeros(image.shape[:2], np.uint8)                                             
            mask_in_origin[t:b,l:r] = mask 
            mask = mask.astype(np.uint8)
            # append to lists
            masks_in_partition.append(mask_in_origin)                 
            all_masks = np.dstack([all_masks, mask_in_origin])

            # save data of this mask, even if it overlaps with a previous mask....
            save_folder = os.path.join(save_root, 'raw_masks')
            os.makedirs(save_folder, exist_ok=True)          
            mask_name = '{};({},{},{},{}).png'.format(part_idx, t,b,l,r)
            save_path = os.path.join(save_folder, mask_name)
            mask = mask.astype(int)*255
            skimage.io.imsave(save_path, mask)

            # add class id to data
            all_ids = np.append(all_ids, p_detections['class_ids'][roi_idx]).astype(int)

            # save a copy of masks which DO NOT overlap
            save_folder = os.path.join(save_root, 'masks')
            os.makedirs(save_folder, exist_ok=True)   
            save_path = os.path.join(save_folder, mask_name)
            overlapping = False  
            if 0.4*(np.sum(mask)/255) > np.sum(np.ones(mask.shape[:2], np.uint8)) :
                # very large crystal is probably a false positive. Ignore.
                continue
            if part_idx == 0:
                # save everything in the first partition       
                print('saving mask', save_path)
                skimage.io.imsave(save_path, mask.astype(np.uint8)) 
            else:
                # this compares all masks with those in neighbouring partitions. (those that have already been analysed)
                # mask is saved if overlap is less than 10%              
                for idx in indexes_to_check:
                    masks_to_check = masks_per_partition[idx]
                    for mask_to_check in masks_to_check:
                        # see which pixels are present in both masks, i.e. the overlapping region
                        mask_overlap = np.bitwise_and(mask_in_origin, mask_to_check)
                        if np.sum(mask_overlap) > np.sum(mask_in_origin)*0.1:                           
                            # if the overlapping region accounts for more than 10% of the original crystal
                            # we decide that the crystals do not overlap. i.e arise from different crystals
                            overlapping=True
                            break
                    # we stop checking if we already know that the crystals are overlapping
                    if overlapping:
                        break
            # if the crystals are not overlapping then we save         
            if not overlapping:
                print('saving image: {}'.format(save_path))
                skimage.io.imsave(save_path, mask.astype(np.uint8))  


        # add all masks from the partition to a list once the partition has been assessed. this is for 
        # checking previous partitions for overlaps.                 
        masks_per_partition.append(masks_in_partition)
    # once all partitions have been assessed...
    if not overwrite: # if we don't want to overwrite....
        if os.path.exists(save_path): # and the path already exists...
            return          # exit function
    # create new axes to display images
    fig, ax = plt.subplots(figsize = (int(image_height/screen_dpi), int(image_width/screen_dpi)))
    # new image will have all detections from each partition stitched together
    visualize.display_instances(image, all_rois, all_masks, all_ids, ['BG', ''], ax = ax)
    # save it
    save_name = '{};[{},{},{}].png'.format(image_title, *partitioning)
    save_path = os.path.join(save_root, save_name)
    plt.savefig(save_path)
    print('saving new stitched image: ', save_path)
    # tidy up pyplot
    plt.clf()
    plt.close('all')
    ax.clear()
    

def run_exp(exp_path, partitioning=[1,1,0], reaction_range=[0,-1], image_range=[0,-1], overwrite=False):
    start, end = reaction_range
    rxns = sorted([i for i in os.listdir(exp_path) if 'reaction' in i.lower()])
    for rxn in rxns[start:end]:
        rxn_path = os.path.join(exp_path, rxn)
        run_rxn(rxn_path, partitioning, image_range, overwrite)
    
def run_rxn(rxn_path, partitioning=[1,1,0], image_range=[0,-1], overwrite=False):
    start, end = image_range
    image_dir = os.path.join(rxn_path, 'Images')
    image_names = sorted([i for i in os.listdir(image_dir)])
    for name in image_names[start: end]:      
        image_path= os.path.join(image_dir, name)
        print(image_path)
        data = detect_path(image_path, partitioning, overwrite)


# run_exp(exp_path, partitioning, reaction_range, image_range, overwrite)
here = os.getcwd()
img_path = os.path.join(here, 'detecting', 'Image_016.png')

detect_path(img_path, partitioning=[10,10,0.2], overwrite=False)