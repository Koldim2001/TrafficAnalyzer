from kafka import KafkaProducer
from json import dumps
import time

from utils_local.utils import profile_time
from elements.VideoEndBreakElement import VideoEndBreakElement
from elements.FrameElement import FrameElement
import logging


class KafkaProducerNode:
    def __init__(self, config) -> None:
        config_kafka = config["kafka_producer_node"]
        bootstrap_servers = config_kafka["bootstrap_servers"]
        self.topic_name = config_kafka["topic_name"]
        self.how_often_sec = config_kafka["how_often_sec"]
        self.camera_id = config_kafka["camera_id"]
        self.last_send_time = None
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda x: dumps(x).encode("utf-8"),
        )

        self.buffer_analytics_sec = (
            config["general"]["buffer_analytics"] * 60 + config["general"]["min_time_life_track"]
        )  # столько по времени буфер набирается и информацию о статистеке выводить рано

    @profile_time
    def process(self, frame_element: FrameElement):
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element

        current_time = time.time()
        timestamp = frame_element.timestamp

        if frame_element.frame_num == 1:
            self.last_send_time = current_time

        if current_time - self.last_send_time > self.how_often_sec or frame_element.frame_num == 1:
            data = {
                f"cars_{self.camera_id}": frame_element.info["cars_amount"],
                f"road_1_{self.camera_id}": (
                    frame_element.info["roads_activity"][1]
                    if timestamp >= self.buffer_analytics_sec
                    else None
                ),
                f"road_2_{self.camera_id}": (
                    frame_element.info["roads_activity"][2]
                    if timestamp >= self.buffer_analytics_sec
                    else None
                ),
                f"road_3_{self.camera_id}": (
                    frame_element.info["roads_activity"][3]
                    if timestamp >= self.buffer_analytics_sec
                    else None
                ),
                f"road_4_{self.camera_id}": (
                    frame_element.info["roads_activity"][4]
                    if timestamp >= self.buffer_analytics_sec
                    else None
                ),
                f"road_5_{self.camera_id}": (
                    frame_element.info["roads_activity"][5]
                    if timestamp >= self.buffer_analytics_sec
                    else None
                ),
            }
            self.kafka_producer.send(self.topic_name, value=data).get(timeout=1)
            logging.info(f"KAFKA sent message: {data} topic {self.topic_name}")
            self.last_send_time = current_time
            frame_element.send_to_kafka = True

        return frame_element
