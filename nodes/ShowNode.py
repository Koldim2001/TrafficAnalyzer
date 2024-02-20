import cv2
import numpy as np
from utils_local.utils import profile_time, FPS_Counter

from elements.FrameElement import FrameElement


class ShowNode:
    def __init__(self, config) -> None:
        config_show_node = config["show_node"]

        self.scale = config_show_node["scale"]
        self.fps_counter_N_frames_stat = config_show_node["fps_counter_N_frames_stat"]
        self.default_fps_counter = FPS_Counter(self.fps_counter_N_frames_stat)
        self.draw_fps_info = config_show_node["draw_fps_info"]
        self.show_roi = config_show_node["show_roi"]
        self.graph_pose = config_show_node["graph_pose"]
        self.imshow = config_show_node["imshow"]

        self.fontFace = 1
        self.fontScale = 2.0
        self.thickness = 2

    @profile_time
    def process(self, frame_element: FrameElement, fps_counter=None):

        frame_result = frame_element.frame.copy()

        # Построение полигонов
        for road_id, points in frame_element.roads_info.items():
            points = np.array(points, np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(frame_result, [points], isClosed=True, color=(0, 255, 0), thickness=2)

            
        if self.draw_fps_info:
            fps_counter = (
                fps_counter if fps_counter is not None else self.default_fps_counter
            )
            fps_real = fps_counter.calc_FPS()

            text = f"FPS: {fps_real:.1f}"
            (label_width, label_height), _ = cv2.getTextSize(
                text,
                fontFace=self.fontFace,
                fontScale=self.fontScale,
                thickness=self.thickness,
            )

            frame = cv2.rectangle(
                frame_result, (0, 0), (10 + label_width, 35 + label_height), (0, 0, 0), -1
            )
            cv2.putText(
                img=frame_result,
                text=text,
                org=(10, 40),
                fontFace=self.fontFace,
                fontScale=self.fontScale,
                thickness=self.thickness,
                color=(255, 255, 255),
            )

        frame_element.frame_result = frame_result
        frame_show = cv2.resize(frame_result.copy(), (-1, -1), fx=self.scale, fy=self.scale)

        if self.imshow:
            cv2.imshow(frame_element.source, frame_show)
            cv2.waitKey(1)
        
        return frame_element

