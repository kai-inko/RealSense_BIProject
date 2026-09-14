import cv2
import numpy as np
from camera_base import CameraBase

class LaptopCamera(CameraBase):
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def start(self):
        if not self.cap.isOpened():
            raise RuntimeError("Could not open Laptop Camera")

    def get_frames(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None, None, None
        
        # Laptop cams don't have real depth or intrinsics object
        # We return None for depth
        return True, frame, None, None

    def stop(self):
        self.cap.release()