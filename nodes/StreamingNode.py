
from torio.io import StreamingMediaEncoder as StreamWriter
import torch
import numpy as np
import time
import cv2
import logging

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
        self.last_send_time = None

        encoder_format = 'yuv420p' if self.encoder == 'libx264' else 'rgb0' # формат пикселей зависит от кодека (gpu или cpu)
        self.device = None if self.encoder == 'libx264' else 'cuda:0'       # выбираем девайс cuda для gpu-кодеков

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
            hw_accel=self.device, 
            encoder_format=encoder_format,
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

        current_time = time.time()
        if frame_element.frame_num == 1:
            self.last_send_time = current_time
        
        if current_time - self.last_send_time >= (1 / (self.fps * 2)):
            frame_result = frame_element.frame_result.copy()
            frame_result = cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB) 
            frame_result = cv2.resize(frame_result, (self.width, self.height))

            # numpy.array -> torch.tensor:
            if isinstance(frame_result, np.ndarray):
                img = torch.from_numpy(frame_result).permute(2, 0, 1).unsqueeze(0)
            else:
                assert False

            #+ Перемещаем тензор на выбранный девайс (CPU или GPU):
            if self.device is None:
                img = img.to('cpu')
            else:
                img = img.to(self.device)
            
            # Подаем очередной кадр в стример:
            try:
                self.stream_writer.write_video_chunk(0, img)
                logging.info(f"Frame was sent to streem {self.output_rtmp}")
                self.last_send_time = current_time
            except:
                logging.error(f"FAILED sending frame to {self.output_rtmp}")

        return frame_element

