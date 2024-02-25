import cv2
from elements.FrameElement import FrameElement
from elements.TrackElement import TrackElement
from utils_local.utils import profile_time, intersects_central_point
import numpy as np


class TrackerInfoUpdateNode:
    """Модуль обновления актуальных треков"""

    def __init__(self, config: dict) -> None:
        config_general = config["general"]

        self.size_buffer_analytics = config_general["buffer_analytics"] * 60  # число секунд в буфере аналитики 
        self.buffer_tracks = {}  # Буфер актуальных треков

    @profile_time 
    def process(self, frame_element: FrameElement) -> FrameElement:
        id_list = frame_element.id_list

        for i, id in enumerate(id_list):
            # Обновление или создание нового трека
            if id not in self.buffer_tracks:
                # Создаем новый ключ
                self.buffer_tracks[id] = TrackElement(
                    id=id,
                    timestamp_first=frame_element.timestamp,
                )
            else:
                # Обновление времени последнего обнаружения
                self.buffer_tracks[id].update(frame_element.timestamp)

            # Поиск первого пересечения с полигонами дорог
            if self.buffer_tracks[id].start_road is None:
                self.buffer_tracks[id].start_road = intersects_central_point(
                    tracked_xyxy=frame_element.tracked_xyxy[i],
                    polygons=frame_element.roads_info,
                )
        
        frame_element.buffer_tracks = self.buffer_tracks

        return frame_element

