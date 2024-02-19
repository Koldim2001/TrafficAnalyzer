from ultralytics import YOLO
import cv2
import torch
import numpy as np
from utils_local.utils import profile_time
from collections import deque

from elements.FrameElement import FrameElement


class DetectionNode:
    def __init__(self, config) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Детекция будет производиться на {device}')
        self.model = YOLO(config["weight_pth"])
        self.model.fuse()
        self.classes = self.model.names
        self.conf = config["confidence"]
        self.iou = config["iou"]
        self.imgsz = config["imgsz"]
        self.classes_to_detect = config["classes_to_detect"]


    @profile_time
    def process(self, frame_element: FrameElement):
        frame = frame_element.frame.copy()

        outputs = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False,
                                     iou=self.iou, classes=self.classes_to_detect)

        frame_element.detected_conf = outputs[0].boxes.conf.cpu().tolist()
        frame_element.detected_cls = outputs[0].boxes.cls.cpu().int().tolist()
        frame_element.detected_xyxy = outputs[0].boxes.xyxy.cpu().int().tolist()

        return frame_element

