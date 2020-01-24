
import os
import json
import cv2
import numpy as np

object_folder = r'U:\Clusters\RandomMOF\MaskRCNN\training\datasets\MOF5_single'

labelled_folder = os.path.join(object_folder, 'labelled')
originals_folder  = os.path.join(object_folder, 'original')
val_folder = os.path.join(object_folder, 'val')
train_folder = os.path.join(object_folder, 'train')

data_file = os.path.join(train_folder, 'via_region_data.json')



data = json.load(open(data_file))


def label_train_folder(object_folder):
    if not os.path.exists(object_folder):
        raise 'folder: {} does not exist'.format(object_folder)
    os.makedirs(labelled_folder, exist_ok=True)
    for key, value in data.items():
        filepath = os.path.join(train_folder, value['filename'])
        image = cv2.imread(filepath)
        for region in value['regions']:
            pts = np.asarray([i for i in zip(region['shape_attributes']['all_points_x'], region['shape_attributes']['all_points_y'])])
            cv2.polylines(image,[pts],True,(0,255,255), 2)
        dst_filepath = os.path.join(labelled_folder, value['filename'])
        cv2.imwrite(dst_filepath,image)


def label_orginals_folder(object_folder):
    if not os.path.exists(object_folder):
        raise 'folder: {} does not exist'.format(object_folder)
    image_paths = []
    for key, value in data.items():

        chunk_filepath = os.path.join(labelled_folder, value['filename'])
        image_chunk = cv2.imread(chunk_filepath)
        image_name, roi = value['filename'].split(';')
        roi, ext = roi.split('.')

        original_image_path = os.path.join(originals_folder, image_name+'.'+ext)
        if original_image_path not in image_paths:
            image_paths.append(original_image_path)
            original_image = cv2.imread(original_image_path)
            new_blank = np.zeros(original_image.shape[:3], np.uint8)

        width, height = roi.split(')(') 
        left, right = [int(i) for i in width[1:].split('-')]
        top, bottom = [int(i) for i in height[:-1].split('-')]

        new_blank[left:right, top:bottom] = image_chunk
        print(type(bottom), type(new_blank.shape[0]))
        if (right + (right-left)) > new_blank.shape[0] and (bottom + (bottom-top)) > new_blank.shape[1]:
            print(image_name)
            target_filepath = os.path.join(originals_folder, '{}_labelled.jpg'.format(image_name))
            cv2.imwrite(target_filepath, new_blank)


label_train_folder(object_folder)
# label_orginals_folder(object_folder)