
from torio.io import StreamingMediaEncoder as StreamWriter
import torch
import numpy as np
import logging
import cv2

from elements.VideoEndBreakElement import VideoEndBreakElement
from elements.FrameElement import FrameElement
from utils_local.utils import profile_time


class StreamingNode:
    """Модуль отвечающий, за стриминг выходного RTMP потока"""

    def __init__(self, config) -> None:
        config_stream = config["streaming_node"]
        self.output_rtmp = config_stream["output_rtmp"]
        self.fps = config_stream["fps"]
        self.height = config_stream["height"]
        self.width = config_stream["width"]
        self.encoder = config_stream["encoder"]

        assert self.encoder == 'libx264' or self.encoder == 'h265' # только CPU-кодеки
        
        # Инициализируем стример из torio (torchaudio):
        self.stream_writer = StreamWriter(dst=self.output_rtmp, format="flv")
        
        # Параметры стриминга:
        self.width, self.height = int(self.width), int(self.height)

        # Добавляем выходной видеострим:
        self.stream_writer.add_video_stream(
            frame_rate=int(self.fps), 
            height=self.height, 
            width=self.width,
            encoder=self.encoder, 
            encoder_format='yuv420p'
        )

        # Open stream:
        self.stream_writer.open()

    @profile_time
    def process(self, frame_element: FrameElement) -> FrameElement:
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element
        assert isinstance(
            frame_element, FrameElement
        ), f"StreamingNode | Неправильный формат входного элемента {type(frame_element)}"

        frame_result = frame_element.frame_result.copy()
        frame_result = cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB) 

        # numpy.array -> torch.tensor:
        if isinstance(frame_result, np.ndarray):
            img = torch.from_numpy(frame_result).permute(2, 0, 1).unsqueeze(0)
        else:
            assert False
        
        # Подаем очередной кадр в стример:
        self.stream_writer.write_video_chunk(0, img)

        return frame_element

