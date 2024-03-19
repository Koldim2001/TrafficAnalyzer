from pathlib import Path
import os
import logging
import cv2

from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement

logger = logging.getLogger(__name__)


class VideoSaverNode:
    """Модуль для сохранения видеопотока"""

    def __init__(self, config: dict) -> None:
        self.fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        self.fps = config["fps"]
        self.out_folder = config["out_folder"]
        self._cv2_writer = None

    def process(self, frame_element: FrameElement) -> None:
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            print(f"Видео сохранено в папке {self.out_folder}")
            return
        assert isinstance(
            frame_element, FrameElement
        ), f"VideoSaverNode | Неправильный формат входного элемента {type(frame_element)}"

        source = frame_element.source
        frame = frame_element.frame_result

        if frame is not None:
            out_file_name = source

            if self._cv2_writer is None:
                self._init_cv2_writer(
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                    out_file_name=out_file_name,
                    fps=self.fps,
                )

            self._cv2_writer.write(frame)

    def _init_cv2_writer(
        self, frame_width: int, frame_height: int, out_file_name: str, fps: float
    ) -> None:
        """Инициализирует cv2.VideoWriter для записи файла в нужном разрешении file_extention:

        Args:
            frame_width (int): ширина кадра записываемого видео.
            frame_height (int): высота кадра записываемого видео.
            out_file_name (str): источник обрабатываемого видео
                (для формирования названия записывааемого видео).
            fps (float): количество кадров в секунду записываемого видео.
        """
        out_file_name = os.path.basename(out_file_name)
        Path(self.out_folder).mkdir(parents=True, exist_ok=True)
        save_path = f"{self.out_folder}/{out_file_name}"
        self._cv2_writer = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            fps,
            (frame_width, frame_height),
        )
        logger.info(f"Saving out video in {save_path}")
