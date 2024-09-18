import numpy as np
import tritonclient.http as httpclient
import cv2
from utils_local.utils import *


def infer_triton_yolo(
    triton_client,
    triton_model_name,
    image_bgr,
    imgsz,
    classes,
    conf,
    iou,
    input_name="images",
    output_name="output0",
):
    orig_width = image_bgr.shape[1]
    scale = orig_width / imgsz
    image_bgr, y_offset = letterbox_resize(image_bgr, (imgsz, imgsz))
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = np.array(image).astype(np.float32) / 255
    image = np.transpose(image, (2, 0, 1))  # Convert to CHW format
    image = np.expand_dims(image, axis=0)

    inputs = [httpclient.InferInput(input_name, image.shape, "FP32")]
    inputs[0].set_data_from_numpy(image)
    outputs = [httpclient.InferRequestedOutput(output_name)]
    output = triton_client.infer(triton_model_name, inputs, outputs=outputs)
    output = output.as_numpy(output_name)[0].T

    # ----------------------------------- постобработка ------------------------------------

    bboxes, classes, confs = [], [], []

    for detection in output:
        classes_scores = detection[4:]
        cls_indx = int(np.argmax(classes_scores))
        confidence = classes_scores[cls_indx]
        if confidence > conf and cls_indx in classes:
            center_x = detection[0]
            center_y = detection[1]
            width = detection[2]
            height = detection[3]

            x1 = int((center_x - width / 2) * scale)
            y1 = int((center_y - height / 2 - y_offset) * scale)
            x2 = int((center_x + width / 2) * scale)
            y2 = int((center_y + height / 2 - y_offset) * scale)

            bboxes.append([x1, y1, x2, y2])
            classes.append(cls_indx)
            confs.append(confidence)

    # Применение NMS
    indeces = select_nms(bboxes, confs, classes, iou_threshold=iou, agnostic=False)

    # Фильтрация боксов, классов и доверий по индексам из NMS
    filtered_bboxes = [bboxes[i] for i in indeces]
    filtered_classes = [classes_list[i] for i in indeces]
    filtered_confs = [confs[i] for i in indeces]

    return filtered_bboxes, filtered_classes, filtered_confs