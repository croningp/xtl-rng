import cv2
import os
import sys

SCAPA_ROOT = '/mnt/scapa4'
ORKNEY_ROOT  ='/mnt/orkney1'

if sys.platform == 'win32':
    SCAPA_ROOT = os.path.join('Z:', os.sep, 'group')
    ORKNEY_ROOT = os.path.join('U:', os.sep)
    
ORKNEY_PARTITIONED = os.path.join(ORKNEY_ROOT, 'Clusters', 'RandomXtl', 'images', 'partitioned')

IMAGE_EXTS = ['png', 'jpg']

def partition_image(image_path, category='', subcategory='', rows=1, cols=1, overlapping=0):
              
        # get path info about input image
        image_name, image_ext = os.path.basename(image_path).split('.')

        this_image_dir = os.path.join(ORKNEY_PARTITIONED, category, subcategory, image_name)
        os.makedirs(this_image_dir, exist_ok=True)

         # load image and specify roi parameters based on input
        image = cv2.imread(image_path)
        cv2.imwrite(os.path.join(this_image_dir, os.path.basename(image_path)), image)
        # use dimensions of original image and input parameters to set variables for selecting roi
        image_height, image_width = image.shape[:2]
        roi_width = int(image_width/cols)
        roi_height = int(image_height/rows)
        overlap_width = int(roi_width*overlapping)
        overlap_height = int(roi_height*overlapping)
                
        #this section first chooses appropriate region of interest, then detects in each region,
        #then restitches the orginal image with detections
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
   
                roi = image[top2:bottom2, left2:right2]

                # name contains 'image title';'roi'
                save_dir = os.path.join(this_image_dir, '[{},{}],[{}]'.format(rows, cols, overlapping))
                os.makedirs(save_dir, exist_ok=True)
                new_image_name ='{};({}-{})({}-{}).{}'.format(image_name,top2, bottom2, left2, right2, image_ext)
                # save folder contains 'chunking settings', 'overlapping settings'           
                save_path = os.path.join(save_dir, new_image_name)
                print(save_path)
                cv2.imwrite(save_path, roi)  

               
        
def partition_folder(folder, category='', subcategory='', rows=1, cols=1, overlapping=0):
    names = [i for i in os.listdir(folder) if i[-3:].lower() in IMAGE_EXTS]
    for name in names:
        image_path = os.path.join(folder, name)
        partition_image(image_path, category, subcategory,rows, cols, overlapping)


if __name__ == '__main__':
    input_path = sys.argv[1]
    category = sys.argv[2]
    subcategory = sys.argv[3]
    rows = int(sys.argv[4])
    cols = int(sys.argv[5])
    overlapping = float(sys.argv[6])
    if os.path.isfile(input_path):
        partition_image(input_path, category, subcategory, rows, cols, overlapping)
    else:
        partition_folder(input_path, category, subcategory, rows, cols, overlapping)





