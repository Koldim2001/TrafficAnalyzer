from ultralytics import YOLO
import torch
import numpy as np

from utils_local.utils import profile_time
from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement
from byte_tracker.byte_tracker_model import BYTETracker as ByteTracker


class DetectionTrackingNodes:
    """Модуль инференса модели детекции + трекинг алгоритма"""

    def __init__(self, config) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Детекция будет производиться на {device}')

        config_yolo = config["detection_node"]
        self.model = YOLO(config_yolo["weight_pth"], task='detect')
        self.classes = self.model.names
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

        outputs = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False,
                                     iou=self.iou, classes=self.classes_to_detect)

        frame_element.detected_conf = outputs[0].boxes.conf.cpu().tolist()
        detected_cls = outputs[0].boxes.cls.cpu().int().tolist()
        frame_element.detected_cls = [self.classes[i] for i in detected_cls]
        frame_element.detected_xyxy = outputs[0].boxes.xyxy.cpu().int().tolist()

        # Преподготовка данных на подачу в трекер
        detections_list = self._get_results_dor_tracker(outputs)

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

    def _get_results_dor_tracker(self, results) -> np.ndarray:
        # Приведение данных в правильную форму для трекера
        detections_list = []
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            # трекаем те же классы что и детектируем
            if class_id[0] in self.classes_to_detect:

                bbox = result.boxes.xyxy.cpu().numpy()
                confidence = result.boxes.conf.cpu().numpy()

                class_id_value = (
                    2  # Будем все трекуемые объекты считать классом car чтобы не было ошибок
                )

                merged_detection = [
                    bbox[0][0],
                    bbox[0][1],
                    bbox[0][2],
                    bbox[0][3],
                    confidence[0],
                    class_id_value,
                ]

                detections_list.append(merged_detection)

        return np.array(detections_list)
