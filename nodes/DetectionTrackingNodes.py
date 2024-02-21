from ultralytics import YOLO
import cv2
import torch
import numpy as np
from utils_local.utils import profile_time
from collections import deque

from elements.FrameElement import FrameElement
from byte_tracker.byte_tracker_model import BYTETracker as ByteTracker



class DetectionTrackingNodes:
    def __init__(self, config) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Детекция будет производиться на {device}')

        config_yolo = config["detection_node"]
        self.model = YOLO(config_yolo["weight_pth"])
        self.model.fuse()
        self.classes = self.model.names
        self.conf = config_yolo["confidence"]
        self.iou = config_yolo["iou"]
        self.imgsz = config_yolo["imgsz"]
        self.classes_to_detect = config_yolo["classes_to_detect"]

        config_bytetrack= config["tracking_node"]

        # ByteTrack param
        first_track_thresh = config_bytetrack["first_track_thresh"]
        second_track_thresh = config_bytetrack["second_track_thresh"]
        match_thresh = config_bytetrack["match_thresh"]
        track_buffer = config_bytetrack["track_buffer"]
        fps = 30  # ставим равным 30 чтобы track_buffer мерился в кадрах
        self.tracker = ByteTracker(fps, first_track_thresh, second_track_thresh, match_thresh, track_buffer, 1)


    @profile_time
    def process(self, frame_element: FrameElement):
        frame = frame_element.frame.copy()

        outputs = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False,
                                     iou=self.iou, classes=self.classes_to_detect)

        frame_element.detected_conf = outputs[0].boxes.conf.cpu().tolist()
        detected_cls = outputs[0].boxes.cls.cpu().int().tolist()
        frame_element.detected_cls = [self.classes[i] for i in detected_cls]
        frame_element.detected_xyxy = outputs[0].boxes.xyxy.cpu().int().tolist()

        return frame_element

