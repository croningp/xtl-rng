import platform
import os
import sys
import cv2
import csv

import routine




if __name__ == '__main__':
    prefix = 'U:\\' if platform.system() == 'Windows' else '/mnt/orkney1'
    data_root = 'Chemobot\\crystalbot_imgs'
    compound = 'W19'
    experiment = '20180215-0'
    exp_path = os.path.join(prefix, data_root, compound, experiment)

    string = routine.whole_experiment(exp_path)

    with open('test.txt', 'w') as f:
        f.write(string)
