import cv2
import numpy as np
from utils_local.utils import profile_time, FPS_Counter

from elements.FrameElement import FrameElement


class ShowNode:
    def __init__(self, config) -> None:
        data_colors = config["general"]["colors_of_roads"]
        self.colors_roads = {key: tuple(value) for key, value in data_colors.items()}
        
        config_show_node = config["show_node"]
        self.scale = config_show_node["scale"]
        self.fps_counter_N_frames_stat = config_show_node["fps_counter_N_frames_stat"]
        self.default_fps_counter = FPS_Counter(self.fps_counter_N_frames_stat)
        self.draw_fps_info = config_show_node["draw_fps_info"]
        self.show_roi = config_show_node["show_roi"]
        self.overlay_transparent_mask = config_show_node["overlay_transparent_mask"]
        self.graph_pose = config_show_node["graph_pose"]
        self.imshow = config_show_node["imshow"]
        self.show_yolo_detections = False

        self.fontFace = 1
        self.fontScale = 2.0
        self.thickness = 2

    @profile_time
    def process(self, frame_element: FrameElement, fps_counter=None):

        frame_result = frame_element.frame.copy()

        # Отображение результатов детекции:
        if self.show_yolo_detections:
            for box, class_name in zip(frame_element.detected_xyxy, frame_element.detected_cls):
                x1, y1, x2, y2 = box
                # Отрисовка прямоугольника
                cv2.rectangle(frame_result, (x1, y1), (x2, y2), (0, 0, 0), 2)
                # Добавление подписи с именем класса
                cv2.putText(frame_result, class_name, (x1, y1 - 10),
                            fontFace=self.fontFace,
                            fontScale=self.fontScale,
                            thickness=self.thickness,
                            color=(0, 0, 255)
                            )

        else:
            # Отображение результатов трекинга:
            for box, class_name, id in zip(frame_element.tracked_xyxy,
                                       frame_element.tracked_cls,
                                       frame_element.id_list):
                x1, y1, x2, y2 = box
                # Отрисовка прямоугольника
                cv2.rectangle(frame_result, (x1, y1), (x2, y2), (50, 25, 50), 2)
                # Добавление подписи с именем класса
                cv2.putText(frame_result, f'{class_name} {id}', (x1, y1 - 10),
                            fontFace=self.fontFace,
                            fontScale=self.fontScale,
                            thickness=self.thickness,
                            color=(0, 0, 255)
                            )

        # Построение полигонов дорог
        if self.show_roi:
            for road_id, points in frame_element.roads_info.items():
                color = self.colors_roads[int(road_id)]
                points = np.array(points, np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(frame_result, [points], isClosed=True, color=color, thickness=2)
                if self.overlay_transparent_mask:
                    frame_result = self._overlay_transparent_mask(frame_result, points,
                                                                  mask_color=color, alpha=0.5)
                
        # Подсчет fps и отрисовка   
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

    
    def _overlay_transparent_mask(self, img, points, mask_color=(0, 255, 255), alpha=0.3):
        binary_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        binary_mask = cv2.fillPoly(binary_mask, pts=[points], color=1)
        colored_mask = (binary_mask[:, :, np.newaxis] * mask_color).astype(np.uint8)
        return cv2.addWeighted(img, 1, colored_mask, alpha, 0)