import logging
import time
import numpy as np
from shapely.geometry import Point, Polygon

logger_profile = logging.getLogger("profile")


def profile_time(func):
    def exec_and_print_status(*args, **kwargs):
        t_start = time.time()
        out = func(*args, **kwargs)
        t_end = time.time()
        dt_msecs = (t_end - t_start) * 1000

        self = args[0]
        logger_profile.debug(
            f"{self.__class__.__name__}.{func.__name__}, time spent {dt_msecs:.2f} msecs"
        )
        return out

    return exec_and_print_status


class FPS_Counter:
    def __init__(self, calc_time_perion_N_frames: int) -> None:
        """Счетчик FPS по ограниченным участкам видео (скользящему окну).

        Args:
            calc_time_perion_N_frames (int): количество фреймов окна подсчета статистики.
        """
        self.time_buffer = []
        self.calc_time_perion_N_frames = calc_time_perion_N_frames

    def calc_FPS(self) -> float:
        """Производит рассчет FPS по нескольким кадрам видео.

        Returns:
            float: значение FPS.
        """
        time_buffer_is_full = len(self.time_buffer) == self.calc_time_perion_N_frames
        t = time.time()
        self.time_buffer.append(t)

        if time_buffer_is_full:
            self.time_buffer.pop(0)
            fps = len(self.time_buffer) / (self.time_buffer[-1] - self.time_buffer[0])
            return np.round(fps, 2)
        else:
            return 0.0


def intersects_central_point(tracked_xyxy, polygons):
    """Функция определяет присутвие центральной точки bbox в  области полигонов дорог

    Args:
        tracked_xyxy: координаты bbox
        polygons: словарь полигонов

    Returns:
        Лиибо None либо значение ключа (номер дороги - int)
    """
    # Центральная точка bbox:
    center_point = [
        (tracked_xyxy[0] + tracked_xyxy[2]) / 2,
        (tracked_xyxy[1] + tracked_xyxy[3]) / 2,
    ]
    center_point = Point(center_point)
    for key, polygon in polygons.items():
        polygon = Polygon([(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)])
        if polygon.contains(center_point):
            return int(key)
    return None


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def agnostic_nms(boxes, confidences, iou_threshold=0.5):
    """Agnostic Non-Maximum Suppression."""
    boxes = np.array(boxes)
    confidences = np.array(confidences)
    indices = np.argsort(confidences)[::-1]

    selected_indices = []

    while len(indices) > 0:
        current_index = indices[0]
        selected_indices.append(current_index)
        iou = np.array([box_iou(boxes[current_index], boxes[i]) for i in indices[1:]])
        indices = indices[np.where(iou <= iou_threshold)[0] + 1]

    return selected_indices


def non_agnostic_nms(boxes, confidences, classes, iou_threshold=0.5):
    """Non-Agnostic Non-Maximum Suppression with class consideration."""
    boxes = np.array(boxes)
    confidences = np.array(confidences)
    classes = np.array(classes)
    indices = np.argsort(confidences)[::-1]

    selected_indices = []

    while len(indices) > 0:
        current_index = indices[0]
        selected_indices.append(current_index)
        current_class = classes[current_index]
        iou = np.array([box_iou(boxes[current_index], boxes[i]) for i in indices])
        class_mask = classes[indices] == current_class
        indices = indices[np.where(np.logical_or(iou <= iou_threshold, ~class_mask))[0]]

    return selected_indices


def select_nms(boxes, confidences, classes=None, iou_threshold=0.5, agnostic=True):
    """Select between Agnostic and Non-Agnostic NMS."""
    if agnostic:
        return agnostic_nms(boxes, confidences, iou_threshold)
    else:
        if classes is None:
            raise ValueError("Classes must be provided for non-agnostic NMS.")
        return non_agnostic_nms(boxes, confidences, classes, iou_threshold)


def letterbox_resize(image, target_size):
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size
    ratio = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    resized_image = cv2.resize(image, (new_width, new_height))
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized_image
    return canvas, y_offset