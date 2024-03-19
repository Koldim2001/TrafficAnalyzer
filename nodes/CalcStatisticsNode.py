from collections import deque
import numpy as np

from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement
from utils_local.utils import profile_time


class CalcStatisticsNode:
    """Модуль для расчета загруженности дорог (вычисление статистик)"""

    def __init__(self, config: dict) -> None:
        config_general = config["general"]

        self.time_buffer_analytics = config_general[
            "buffer_analytics"
        ]  # размер времени буфера в минутах
        self.min_time_life_track = config_general[
            "min_time_life_track"
        ]  # минимальное время жизни трека в сек
        self.count_cars_buffer_frames = config_general["count_cars_buffer_frames"]
        self.cars_buffer = deque(maxlen=self.count_cars_buffer_frames)  # создали буфер значений

    @profile_time 
    def process(self, frame_element: FrameElement) -> FrameElement:
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element
        assert isinstance(
            frame_element, FrameElement
        ), f"CalcStatisticsNode | Неправильный формат входного элемента {type(frame_element)}"

        buffer_tracks = frame_element.buffer_tracks
        self.cars_buffer.append(len(frame_element.id_list))

        info_dictionary = {}
        info_dictionary["cars_amount"] = round(np.mean(self.cars_buffer))
        roads_activity = {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
        }  # всего 5 дорог (занулим стартовое значение)

        # Посчитаем чило машин которые довно живут и имеют значения дороги приезда
        for _, track_element in buffer_tracks.items():
            if (
                track_element.timestamp_last - track_element.timestamp_init_road
                > self.min_time_life_track
                and track_element.start_road is not None
            ):
                key = track_element.start_road
                roads_activity[key] += 1

        # Переведем значения в размерность машин/мин согласно известному размеру буфера
        for key in roads_activity:
            roads_activity[key] /= self.time_buffer_analytics

        info_dictionary['roads_activity'] = roads_activity

        # Запись результатов обработки:
        frame_element.info = info_dictionary

        return frame_element
