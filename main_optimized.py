from time import sleep, time
from multiprocessing import Process, Queue

import hydra
from tqdm import tqdm

from nodes.VideoReader import VideoReader
from nodes.ShowNode import ShowNode
from nodes.VideoSaverNode import VideoSaverNode
from nodes.DetectionTrackingNodes import DetectionTrackingNodes
from nodes.TrackerInfoUpdateNode import TrackerInfoUpdateNode
from nodes.CalcStatisticsNode import CalcStatisticsNode
from nodes.SendInfoDBNode import SendInfoDBNode

from elements.VideoEndBreakElement import VideoEndBreakElement

PRINT_PROFILE_INFO = False


def proc_frame_reader_and_detection(queue_out: Queue, config: dict, time_sleep_start: int):
    sleep_message = f"Система разогревается.. sleep({time_sleep_start})"
    for _ in tqdm(range(time_sleep_start), desc=sleep_message):
        sleep(1)
    video_reader = VideoReader(config["video_reader"])
    detection_node = DetectionTrackingNodes(config)
    for frame_element in video_reader.process():
        ts0 = time()
        frame_element = detection_node.process(frame_element)
        ts1 = time()
        queue_out.put(frame_element)
        if PRINT_PROFILE_INFO:
            print(
                f"PROC_FRAME_READER_AND_DETECTION: {(time()-ts0) * 1000:.0f} ms: "
                + f"detection_node {(ts1-ts0) * 1000:.0f} | "
                + f"put {(time()-ts1) * 1000:.0f}"
            )
        if isinstance(frame_element, VideoEndBreakElement):
            break


def proc_tracker_update_and_calc(queue_in: Queue, queue_out: Queue, config: dict):
    tracker_info_update_node = TrackerInfoUpdateNode(config)
    calc_statistics_node = CalcStatisticsNode(config)
    send_info_db = config["pipeline"]["send_info_db"]
    if send_info_db:
        send_info_db_node = SendInfoDBNode(config)
    while True:
        ts0 = time()
        frame_element = queue_in.get()
        ts1 = time()
        frame_element = tracker_info_update_node.process(frame_element)
        frame_element = calc_statistics_node.process(frame_element)
        if send_info_db:
            frame_element = send_info_db_node.process(frame_element)
        ts2 = time()
        queue_out.put(frame_element)
        if PRINT_PROFILE_INFO:
            print(
                f"PROC_TRACKER_UPDATE_AND_CALC: {(time()-ts0) * 1000:.0f} ms: "
                + f"get {(ts1-ts0) * 1000:.0f} | "
                + f"nodes_inference {(ts2-ts1) * 1000:.0f} | "
                + f"put {(time()-ts2) * 1000:.0f}"
            )
        if isinstance(frame_element, VideoEndBreakElement):
            break


def proc_show_node(queue_in: Queue, config: dict):
    show_node = ShowNode(config)
    video_saver_node = VideoSaverNode(config["video_saver_node"])
    save_video = config["pipeline"]["save_video"]
    while True:
        ts0 = time()
        frame_element = queue_in.get()
        ts1 = time()
        frame_element = show_node.process(frame_element)
        if save_video:
            video_saver_node.process(frame_element)
        ts2 = time()
        if PRINT_PROFILE_INFO:
            print(
                f"PROC_SHOW_NODE: {(time()-ts0) * 1000:.0f} ms: "
                + f"get {(ts1-ts0) * 1000:.0f} | "
                + f"show_node {(ts2-ts1) * 1000:.0f} | "
                + f"put {(time()-ts2) * 1000:.0f}"
            )
        if isinstance(frame_element, VideoEndBreakElement):
            break


@hydra.main(version_base=None, config_path="configs", config_name="app_config")
def main(config) -> None:
    time_sleep_start = 5

    queue_frame_reader_and_detect_out = Queue(maxsize=50)
    queue_track_update_out = Queue(maxsize=50)

    processes = [
        Process(
            target=proc_frame_reader_and_detection,
            args=(queue_frame_reader_and_detect_out, config, time_sleep_start),
            name="proc_frame_reader_and_detection",
        ),
        Process(
            target=proc_tracker_update_and_calc,
            args=(queue_frame_reader_and_detect_out, queue_track_update_out, config),
            name="proc_tracker_update_and_calc",
        ),
        Process(
            target=proc_show_node,
            args=(queue_track_update_out, config),
            name="proc_show_node",
        ),
    ]

    for p in processes:
        p.daemon = True
        p.start()

    # Ждем, пока последний процесс завершится
    processes[-1].join()


if __name__ == "__main__":
    ts = time()
    main()
    print(f"\n total time: {(time()-ts) / 60:.2} minute")
