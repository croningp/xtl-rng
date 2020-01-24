import os
import cv2
import sys
import hashlib
import numpy as np
import scipy.special as sp
import matplotlib
import matplotlib.pyplot as plt
import datetime  

import utils

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
ORKNEY_VIALS = os.path.join(ORKNEY_CV, 'vials')

ORKNEY_RANDOM = os.path.join(ORKNEY_PROJECT, 'random numbers')


IMG_EXTS = ['.img', '.bmp', '.tiff', '.jpg', '.png']

data = []

def lin2win(filepath):
    return filepath.replace('/mnt/scapa4', 'Z:').replace('/mnt/orkney1', 'U:').replace('/', '\\')

def win2lin(filepath):
    return filepath.replace('Z:', '/mnt/scapa4' ).replace('U:', '/mnt/orkney1').replace('\\', '/')

### calculates the binary sequence

def get_radial_bin_templates(number_of_bins):

    area_per_bin = 1/number_of_bins    
    radial_bin_edges = []
    for idx in range(1, number_of_bins+1):
        area_within_radius = area_per_bin*idx
        radius = np.sqrt(area_within_radius)
        radial_bin_edges.append(radius)   
    return radial_bin_edges

def get_angular_bin_templates(number_of_bins):
    bin_angle = 360/number_of_bins
    angle_bin_edges = []
    for idx in range(1, number_of_bins+1):
        angle_bin_edges.append(bin_angle*idx)
    return angle_bin_edges

def get_radial_bin_masks(vial, radial_bin_templates, master_bg):
    prev_bin_mask = master_bg.copy()
    radial_bins = []
    for edge in radial_bin_templates:
        bg = master_bg.copy()
        cv2.circle(bg, (vial['x'], vial['y']), int(edge*vial['r']), 255, -1)
        radial_bin = bg - prev_bin_mask
        prev_bin_mask = bg
        radial_bins.append(radial_bin)
    return radial_bins

def get_angular_bin_masks(vial, angular_bin_templates, master_bg):
    prev_angle_edge = 0
    angular_bins = []
    angle_size = angular_bin_templates[0]
    for angle_edge in angular_bin_templates:
        bg = master_bg.copy()
        cv2.ellipse(bg, (vial['x'], vial['y']), (vial['r'],vial['r']), 0, prev_angle_edge, angle_edge, 255, -1)
        prev_angle_edge = angle_edge
        angular_bins.append(bg)    
    return angular_bins

def determine_bin(xtl, bin_masks):

    for idx, mask in enumerate(bin_masks):
        if mask[xtl['x'], xtl['y']]:
            return idx

def int2bits(integer, number_of_bits=4):
    return bin(integer)[2:].zfill(number_of_bits)

def get_image_data(image_path, version, partitioning, master_bg):

    partition_string = str(partitioning).replace(' ', '')
    
    image_head, image_ext = os.path.splitext(image_path)
    image_title = os.path.basename(image_head)

    reaction_path = os.path.dirname(os.path.dirname(image_path))
    rxn_id = os.path.basename(reaction_path)
    exp_path = os.path.dirname(reaction_path)
    exp_id = os.path.basename(exp_path)
    cmpd_path = os.path.dirname(exp_path)
    cmpd_id = os.path.basename(cmpd_path)
    image = cv2.imread(image_path)

    output_dir = os.path.join(ORKNEY_RANDOM, 'microstates', cmpd_id, exp_id, rxn_id, image_title, 'strings')
    
    vial_mask_path = os.path.join(ORKNEY_VIALS, cmpd_id, exp_id, rxn_id, 'masks', image_title+'.png')
    vial_data_path = os.path.join(ORKNEY_VIALS, cmpd_id, exp_id, rxn_id, 'data', image_title+'.txt')
    
    vial_mask = cv2.imread(vial_mask_path)
    vX, vY, vR = [int(i) for i in open(vial_data_path, 'r').read().split(',')]
    vial = {'x': vX, 'y': vY, 'r':vR}
    crystal_masks_dir = os.path.join(ORKNEY_DETECTED, cmpd_id+version, exp_id, rxn_id, image_title, partition_string, 'good_masks')

    crystal_masks_names = os.listdir(crystal_masks_dir)
   
    xtls = []

    for name in crystal_masks_names:
        mask_path = os.path.join(crystal_masks_dir, name)

        partition_idx, bbox = name[:-4].split(';')
        t,b,l,r = [int(i) for i in bbox[1:-1].split(',')]

        mask = cv2.imread(mask_path, 0)
        findcounteridx = 1
        if os.name == 'Posix': findcounteridx = 0
        try:
            cnt = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[findcounteridx][0]
            area = cv2.contourArea(cnt)
            img_mask = master_bg.copy()
            img_mask[t:b,l:r] = mask
            raw_mask = cv2.bitwise_and(image, image, mask=img_mask)
            name = os.path.split(os.path.splitext(mask_path)[0])[-1]
            col_roi = raw_mask[t:b,l:r]
            if area > 10:
                ellipse = cv2.fitEllipse(cnt)

                xtl = {'path':mask_path, 'name':name, 'partitioning':partitioning,
                        'partition idx':int(partition_idx), 'bbox': [t,b,l,r],
                        't': t, 'b': b, 'l': l, 'r': r, 'w': b-t, 'h': r-l,'area': area,
                        'x': int((l+r)/2), 'y': int((t+b)/1),
                        'bbox area':(b-t)*(r-l), 'x': int((b+t)/2), 'y': int((r+l)/2),
                        'mask': mask, 'cnt':cnt, 'ellipse':ellipse,'rotation': ellipse[2],
                        'image mask': img_mask, 'col roi':col_roi, 'name': name
                       }
                xtls.append(xtl)
        except:
            pass

    return xtls, vial, output_dir
    
    
def get_vial_pixels(vial):
    diameter = vial['r']*2
    top = vial['y'] - vial['r']
    left = vial['x'] - vial['r']
    pixels = []

    for y in range(top, top + diameter):
        if y > 0 and y < 800:
            for x in range(left, left+diameter):
                if np.hypot(x-vial['x'],y-vial['y']) < vial['r']:
                    pixels.append([x,y])
                
    
    return pixels

def generate_string(image_path, version, partitioning, bits_per_attribute):
    image = cv2.imread(image_path)
    
    grey_image = cv2.imread(image_path, 0)
    image_name = os.path.splitext(image_path)[0]
    master_bg = np.zeros(image.shape[:2], np.uint8)

    xtls, vial, strings_output_dir = get_image_data(image_path, version, partitioning, master_bg)
    
    os.makedirs(strings_output_dir, exist_ok=True)
    all_output_path = os.path.join(strings_output_dir, 'all.txt')    
    radial_bin_templates = get_radial_bin_templates(bits_per_attribute**2)
    angular_bin_templates = get_angular_bin_templates(bits_per_attribute**2)


    radial_bin_masks = get_radial_bin_masks(vial, radial_bin_templates, master_bg)
    angular_bin_masks = get_angular_bin_masks(vial, angular_bin_templates, master_bg)

    xtls_by_area = sorted(xtls, key=lambda i: i['area'])
  
    sequence = []
    bits = []
    allbits = ''
    mask = np.zeros(image.shape[:2], np.uint8)
    
    prev_area = xtls_by_area[0]['area']
    
    for idx, xtl in enumerate(xtls_by_area):         
        count = 0
        rad_idx = determine_bin(xtl, radial_bin_masks)              
        ang_idx = determine_bin(xtl, angular_bin_masks)
        if rad_idx and ang_idx:
            if xtl['area'] > 4000:
                break
            mask = cv2.bitwise_or(mask, xtl['image mask'])
            prev_area = xtl['area']
    
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    pixels = get_vial_pixels(vial)
    rawstring = ''
    for p in pixels:
        value = masked_img[p[1], p[0]][2]
        pbits = bin(value)[2:].zfill(8)
        rawstring += pbits

    string_sections = [rawstring[i*512:(i+1)*512] for i in range(int(len(rawstring)/512))]
    string = ''
    count = 0
    for section in string_sections:
        if section != '0'*512:
            hash_object = hashlib.sha512(section.encode())
            hex_dig = hash_object.hexdigest()
 
            for j in (hex_dig):
                bits = bin(int(int(j, 16)))[2:].zfill(4)
                string += bits     

    with open(all_output_path, 'w') as f:
        f.write(string)
    return string, strings_output_dir

def run_exp(exp_path, version, partitioning=[4,4,0.2], bits_per_attribute=4, m=0, reaction_range=[0,-1], image_range=[0,-1]):
    start, end = reaction_range
    rxns = sorted([i for i in os.listdir(exp_path) if 'reaction' in i.lower()])
    exp_apens, exp_defs, exp_strings = [], [], []
    for rxn in rxns[start:end]:     
        rxn_path = os.path.join(exp_path, rxn)
        strings, rxn_apens, rxn_defs = run_rxn(rxn_path, version, partitioning, bits_per_attribute, m, image_range)
       
    return exp_strings, exp_apens, exp_defs

def run_rxn(rxn_path, version, partitioning=[4,4,0.2], bits_per_attribute=4, m=0, image_range=[0,-1]):

    start, end = image_range
    image_dir = os.path.join(rxn_path, 'Images')
    image_names = sorted([i for i in os.listdir(image_dir)])
    rxn_apens, rxn_defs, rxn_strings = [], [], []
    final_string = ''
    for name in image_names[start: end]:    
        
        image_path= os.path.join(image_dir, name)
        sys.stdout.write('\r{}'.format(image_path))
        
        string, strings_output_dir = generate_string(image_path, version, partitioning, bits_per_attribute)
        final_string += string
    return final_string


Cu1 = {'path': '/mnt/orkney1/Chemobot/crystalbot_imgs/CuSO4/180906a',
     'ver': '',
     'partitioning':[6,6,0.2],
     'image': [-2,-1],
     'reactions':[0,49]}
Cu2 = {'path': '/mnt/orkney1/Chemobot/crystalbot_imgs/CuSO4/180907d',
     'ver': '',
     'partitioning':[6,6,0.2],
     'image': [31,32],
     'reactions':[0,47]}
W1 = {'path': '/mnt/orkney1/Chemobot/crystalbot_imgs/W19/20180214-1',
     'ver': '',
     'partitioning':[4,4,0.2],
     'image': [78,79],
     'reactions':[0,99]}
W2 = {'path': '/mnt/orkney1/Chemobot/crystalbot_imgs/W19/20180214-0',
     'ver': '',
     'partitioning':[6,6,0.2],
     'image': [20,21],
     'reactions':[0,99]}
Co1 = {'path': '/mnt/orkney1/Chemobot/crystalbot_imgs/Co4/20180216-0',
     'ver': 'v2',
     'partitioning':[6,6,0.2],
     'image': [70,71],
     'reactions':[0,99]}
Co2 = {'path': '/mnt/orkney1/Chemobot/crystalbot_imgs/Co4/20180219-0',
     'ver': 'v2',
     'partitioning':[6,6,0.2],
     'image': [51,52],
     'reactions':[0,99]}

EXPS = [W1]

if __name__ == '__main__':

    bins = 16

    for exp in EXPS:#, Cu, W]:
        exp_path = exp['path']
        version = exp['ver']
        exp_path = lin2win(exp_path)
        partitioning = exp['partitioning']
        reaction_range = exp['reactions']
        image_range = exp['image']

        bits_per_attribute = bins

        exp_strings, exp_apens, exp_defs = run_exp(exp_path, version, partitioning = partitioning, bits_per_attribute=bins,
                                                   m=1, reaction_range=reaction_range, image_range=image_range)

