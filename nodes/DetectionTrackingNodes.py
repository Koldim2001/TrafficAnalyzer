from ultralytics import YOLO
import torch
import numpy as np

from utils_local.utils import profile_time
from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement
from byte_tracker.byte_tracker_model import BYTETracker as ByteTracker
import tritonclient.http as httpclient
from utils_local.infer_triton_utils import infer_triton_yolo


class DetectionTrackingNodes:
    """Модуль инференса модели детекции + трекинг алгоритма"""

    def __init__(self, config) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Детекция будет производиться на {device}')

        config_yolo = config["detection_node"]

        self.triton_client_yolo = httpclient.InferenceServerClient(url=yolo_config["triton_socket"])
        self.triton_model_name_yolo = self.yolo_config["triton_model_name"]

        self.classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"]
        self.conf = config_yolo["confidence"]
        self.iou = config_yolo["iou"]
        self.imgsz = config_yolo["imgsz"]
        self.classes_to_detect = config_yolo["classes_to_detect"]

        config_bytetrack= config["tracking_node"]

        # ByteTrack param
        first_track_thresh = config_bytetrack["first_track_thresh"]
        second_track_thresh = config_bytetrack["second_track_thresh"]
        match_thresh = config_bytetrack["match_thresh"]
        track_buffer = config_bytetrack["track_buffer"]
        fps = 30  # ставим равным 30 чтобы track_buffer мерился в кадрах
        self.tracker = ByteTracker(
            fps, first_track_thresh, second_track_thresh, match_thresh, track_buffer, 1
        )

    @profile_time
    def process(self, frame_element: FrameElement) -> FrameElement:
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element
        assert isinstance(
            frame_element, FrameElement
        ), f"DetectionTrackingNodes | Неправильный формат входного элемента {type(frame_element)}"

        frame = frame_element.frame.copy()

        frame_element.detected_xyxy, detected_cls, frame_element.detected_conf = infer_triton_yolo(
            self.triton_client_yolo,
            self.triton_model_name_yolo,
            frame,
            self.imgsz,
            self.classes_to_detect,
            self.conf,
            self.iou,
        )

        frame_element.detected_cls = [self.classes[i] for i in detected_cls]

        # Преподготовка данных на подачу в трекер
        detections_list = self._get_results_dor_tracker(
            frame_element.detected_xyxy,
            detected_cls,
            frame_element.detected_conf
        )

        # Если детекций нет, то оправляем пустой массив
        if len(detections_list) == 0:
            detections_list = np.empty((0, 6))

        track_list = self.tracker.update(torch.tensor(detections_list), xyxy=True)

        # Получение id list
        frame_element.id_list = [int(t.track_id) for t in track_list]

        # Получение box list
        frame_element.tracked_xyxy = [list(t.tlbr.astype(int)) for t in track_list]

        # Получение object class names
        frame_element.tracked_cls = [self.classes[int(t.class_name)] for t in track_list]

        # Получение conf scores
        frame_element.tracked_conf = [t.score for t in track_list]

        return frame_element

    def _get_results_for_tracker(self, filtered_bboxes, filtered_classes, filtered_confs) -> np.ndarray:
        # Приведение данных в правильную форму для трекера
        detections_list = []
        for bbox, class_id, confidence in zip(filtered_bboxes, filtered_classes, filtered_confs):
            # трекаем те же классы что и детектируем
            if class_id in self.classes_to_detect:
                class_id_value = 2  # Будем все трекуемые объекты считать классом car чтобы не было ошибок

                merged_detection = [
                    bbox[0],  # x1
                    bbox[1],  # y1
                    bbox[2],  # x2
                    bbox[3],  # y2
                    confidence,
                    class_id_value,
                ]

                detections_list.append(merged_detection)

        return np.array(detections_list)