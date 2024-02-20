import numpy as np


class FrameElement:
    def __init__(
        self,
        source: str,
        frame: np.ndarray,
        timestamp: int,
        frame_num: float,
        roads_info: dict,
        frame_result: np.ndarray | None = None,
        detected_conf: list | None = None,
        detected_cls: list | None = None,
        detected_xyxy: list[list] | None = None,
    ) -> None:
        self.source = source  # Путь к видео или номер камеры с которой берем поток
        self.frame = frame  # Кадр bgr формата 
        self.timestamp = timestamp  # Значение времени с начала потока (в секундах)
        self.frame_num = frame_num  # Нормер кадра с потока
        self.roads_info = roads_info  # Словарь с координатми дорог, примыкающих к участку кругового движения
        self.frame_result = frame_result  # Итоговый обработанный кадр
        self.detected_conf = detected_conf  # Список уверенностей задетектированных объектов
        self.detected_cls = detected_cls  # Список классов задетектированных объектов
        self.detected_xyxy = detected_xyxy  # Список списков с координатами xyxy боксов

