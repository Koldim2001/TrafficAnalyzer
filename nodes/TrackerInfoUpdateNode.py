import cv2
from elements.FrameElement import FrameElement
from elements.TrackElement import TrackElement
import numpy as np


class TrackerInfoUpdateNode:
    """Модуль обновления актуальных треков"""

    def __init__(self, config: dict) -> None:
        config_general = config["general"]

        self.size_buffer_analytics = config_general["buffer_analytics"] * 60  # число секунд в буфере аналитики 
        self.buffer_tracks = {}  # Буфер актуальных треков

        
    def process(self, frame_element: FrameElement) -> FrameElement:
        
        frame_element.buffer_tracks = self.buffer_tracks

        return frame_element

