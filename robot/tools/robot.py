import os
import sys
import json
import threading

import cv2

from commanduino import CommandManager
from commanduino.devices.axis import Axis, MultiAxis
from pycont.controller import MultiPumpController
from chemobot_tools.v4l2 import V4L2

PLATFORM_ROOT = sys.path[0]
CONFIG_FOLDER = os.path.join(PLATFORM_ROOT, 'robot', 'configs')

ARDUINO_CONFIG = os.path.join(CONFIG_FOLDER, 'arduino_config.json')
PUMPS_CONFIG = os.path.join(CONFIG_FOLDER, 'pumps_config.json')
CAMERA_CONFIG = os.path.join(CONFIG_FOLDER, 'camera_config.json')
PLATFORM_CONFIG = os.path.join(CONFIG_FOLDER, 'platform_config.json')


class Robot:

    def __init__(self):
        #load the platform dimensions etc
        self.load_platform()

    def load_hardware(self):
        # load each piece of hardware
        self.load_arduino()
        self.load_pumps()
        self.load_camera()

    def load_platform(self):
        # for communication with the platform
        self.plat = Platform(PLATFORM_CONFIG)

    def load_pumps(self):
        # for communication with the pumps
        self.cont = MultiPumpController.from_configfile(PUMPS_CONFIG)
        self.cont.smart_initialize()

    def load_camera(self):
        # for communication with the camera
        self.camera = Camera(0)

    def load_arduino(self):
        # for communication to arduino
        self.cmdMng = CommandManager.from_configfile(ARDUINO_CONFIG)

        # cooling fan
        self.Fan = self.cmdMng.Fan
        # each axis with Arduino communication, movement rate, min position, max position define
        self.X = Axis(self.cmdMng.X, 0.00937, 0, 510)
        self.Y = Axis(self.cmdMng.Y, 0.00937, 0, 270)
        self.Z = Axis(self.cmdMng.Z, 0.00253, 0, 100)
        self.CX = Axis(self.cmdMng.CX, 0.00937, 0, 510)
        self.CY = Axis(self.cmdMng.CY, 0.00937, 0, 270)

        # create multiaxes for concurrent movement
        self.XY = MultiAxis(self.X,self.Y)
        self.CXY = MultiAxis(self.CX,self.CY)
        self.XY_CXY = MultiAxis(self.X,self.Y,self.CX,self.CY)
        self.BOT = MultiAxis(self.X,self.Y,self.Z,self.CX,self.CY)

        self.BOT.home()

class Camera:

    def __init__(self, port):
        self.camera = cv2.VideoCapture(port)
        self.camera.set(3,2000)
        self.camera.set(4,2000)

    def read(self):
        return self.camera.read()

    def start(self):
        self.cam_thread = threading.Thread(target=self.run_cam)
        self.cam_thread.start()

    def run_cam(self):
        while(self.camera.isOpened()):
            ret, frame = self.camera.read()
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def stop(self):
        self.camera.release()
        cv2.destroyAllWindows()
        self.cam_thread.join()

class Platform:

    def __init__(self, PLATFORM_CONFIG):

        #json contains data about the physical dimensions of the platform
        self.configfile = PLATFORM_CONFIG
        self.data = json.load(open(self.configfile))

        self.vial_separation = self.data['vial_separation'] #   in mm
        self.vial_rows = self.data['vial_rows']
        self.vial_columns = self.data['vial_columns']
        self.vial_top = self.data['vial_top'] # depth of the z axis when out of a vial
        self.z_depth = self.data['z_depth']   # depth of the z axis when in a vial

        self.generate_coordinates()

    def generate_coordinates(self):

        self.coords = []

        """ coordinates go in a snake-like pattern """
        for i in range(1,self.vial_rows):
            X = i*self.vial_separation          # x location in mm
            for j in range(self.vial_columns):
                if i % 2 == 1:
                    """ odd rows go forward """
                    Y =  j*self.vial_separation     # y location in mm
                else:
                    """ even rows go backwards """
                    Y =  (self.vial_columns-1-j)*self.vial_separation
                self.coords.append((X, Y))
