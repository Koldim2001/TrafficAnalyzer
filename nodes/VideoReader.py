import os
import json
import time
import logging
from typing import Generator
import cv2

from elements.FrameElement import FrameElement

logger = logging.getLogger(__name__)

class VideoReader:
    """Модуль для чтения кадров с видеопотока"""

    def __init__(self, config: dict) -> None:
        self.video_pth = config["src"]
        self.video_source = f"Processing of {self.video_pth}"
        assert (
            os.path.isfile(self.video_pth) or type(self.video_pth) == int
        ), f"VideoReader| Файл {self.video_pth} не найден"

        self.stream = cv2.VideoCapture(self.video_pth)

        # устанавливаем ширину и высоту при обработке с видео-камеры
        if type(self.video_pth) == int:
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.skip_secs = config["skip_secs"]
        self.last_frame_timestamp = -1  # специально отрицательное при инициализации (костыль)

        # Чтение данных из файла JSON (информация о координатах въезда и выезда дорог)
        with open(config["roads_info"], 'r') as file:
            data_json = json.load(file)

        # Преобразование данных координат дорог в формат int
        self.roads_info = {key: [int(value) for value in values] for key, values in data_json.items()}

    def process(self) -> Generator[FrameElement, None, None]:
        # номер кадра текущего видео
        frame_number = 0

        while True:
            ret, frame = self.stream.read()
            if not ret:
                logger.warning("Can't receive frame (stream end?). Exiting ...")
                break

            if type(self.video_pth) == int:
                timestamp = time.time()
            else:
                timestamp = self.stream.get(cv2.CAP_PROP_POS_MSEC) / 1000

            # делаем костыль, чтобы не было 0-вых тайстампов под конец стрима, баг опенсв
            timestamp = (
                timestamp
                if timestamp > self.last_frame_timestamp
                else self.last_frame_timestamp + 0.1
            )

            if abs(self.last_frame_timestamp - timestamp) < self.skip_secs:
                continue
            self.last_frame_timestamp = timestamp

            frame_number += 1

            yield FrameElement(self.video_source, frame, timestamp, frame_number, self.roads_info)
