import os
import sys
import time
import shutil
import cv2

class Image:

    def __init__(self, experiment):

        self.exp = experiment
        self.coords = self.exp.robot.plat.coords

        self.CXY = self.exp.robot.CXY           # define access to the camera XY motors
        self.camera = self.exp.robot.camera         # define access to camera


    def run(self, reaction_num, image_num):

        self.local_path, self.remote_path = self.exp.filing.prepare_image_filing(reaction_num, image_num)
        self.CXY.move_to(list(self.coords[reaction_num]))

        self.take_image(reaction_num, image_num)

    def take_image(self, reaction_num, image_num):
        for i in range(9):              # loop is to clear the shift register
            ret, image = self.camera.read()
        print('saving image {} for reaction {}'.format(image_num, reaction_num))

        cv2.imwrite(self.local_path, image)
        cv2.imwrite(self.remote_path, image)
