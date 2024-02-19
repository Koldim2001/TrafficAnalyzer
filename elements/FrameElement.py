import numpy as np


class FrameElement:
    def __init__(
        self,
        source: str,
        frame: np.ndarray,
        timestamp: int,
        frame_num: float,
        frame_result: np.ndarray | None = None,
    ) -> None:
        self.source = source  # Путь к видео или номер камеры с которой берем поток
        self.frame = frame  # Кадр bgr формата 
        self.timestamp = timestamp  # Значение времени с начала потока (в секундах)
        self.frame_num = frame_num  # Нормер кадра с потока
        self.frame_result = frame_result  # Итоговый обработанный кадр
