from collections import deque
import numpy as np

from elements.FrameElement import FrameElement
from elements.TrackElement import TrackElement
from utils_local.utils import profile_time

class CalcStatisticsNode:
    """Модуль для расчета загруженности дорог (вычисление статистик)"""

    def __init__(self, config: dict) -> None:
        config_general = config["general"]

        self.min_time_life_track = config_general["min_time_life_track"]
        self.count_cars_buffer_frames = config_general["count_cars_buffer_frames"]
        self.cars_buffer = deque(maxlen=self.count_cars_buffer_frames)  # создали буфер значений
          
    @profile_time 
    def process(self, frame_element: FrameElement) -> FrameElement:
        buffer_tracks = frame_element.buffer_tracks
        self.cars_buffer.append(len(frame_element.id_list))

        info_dictionary = {}
        info_dictionary['cars_amount'] = round(np.mean(self.cars_buffer))

        # Запись результатов обработки:
        frame_element.info = info_dictionary

        return frame_element

