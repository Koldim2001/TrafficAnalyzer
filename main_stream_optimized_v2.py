from time import sleep, time
from multiprocessing import Process, Queue
from queue import Full as queue_is_full

import hydra
from tqdm import tqdm

from nodes.VideoReader import VideoReader
from nodes.ShowNode import ShowNode
from nodes.VideoSaverNode import VideoSaverNode
from nodes.DetectionTrackingNodes import DetectionTrackingNodes
from nodes.TrackerInfoUpdateNode import TrackerInfoUpdateNode
from nodes.CalcStatisticsNode import CalcStatisticsNode
from nodes.FlaskServerVideoNode import VideoServer
from nodes.KafkaProducerNode import KafkaProducerNode
from elements.VideoEndBreakElement import VideoEndBreakElement

PRINT_PROFILE_INFO = False


def proc_frame_reader(queue_out: Queue, config: dict, time_sleep_start: int):
    sleep_message = f"Система разогревается.. sleep({time_sleep_start})"
    for _ in tqdm(range(time_sleep_start), desc=sleep_message):
        sleep(1)
    video_reader = VideoReader(config["video_reader"])
    for frame_element in video_reader.process():
        ts0 = time()
        try:
            queue_out.put_nowait(frame_element)
            
            if PRINT_PROFILE_INFO:
                print(f"PROC_FRAME_READER: {(time()-ts0) * 1000:.0f} ms: ")

        except queue_is_full:
            if PRINT_PROFILE_INFO:
                print("queue_is_full => pass frame")

        if isinstance(frame_element, VideoEndBreakElement):
            break

def proc_proceessor(queue_in: Queue, config: dict, frame_process: Process):
    detection_node = DetectionTrackingNodes(config)
    tracker_info_update_node = TrackerInfoUpdateNode(config)
    calc_statistics_node = CalcStatisticsNode(config)
    send_info_kafka = config["pipeline"]["send_info_kafka"]
    if send_info_kafka:
        kafka_producer_node = KafkaProducerNode(config)
    show_node = ShowNode(config)
    save_video = config["pipeline"]["save_video"]
    show_in_web = config["pipeline"]["show_in_web"]
    if save_video:
        video_saver_node = VideoSaverNode(config["video_saver_node"])
    if show_in_web:
        video_server_node = VideoServer(config)

    while True:

        if not frame_process.is_alive():
            break

        ts0 = time()
        frame_element = queue_in.get()
        ts1 = time()
        frame_element = detection_node.process(frame_element)
        frame_element = tracker_info_update_node.process(frame_element)
        frame_element = calc_statistics_node.process(frame_element)
        if send_info_kafka:
            frame_element = kafka_producer_node.process(frame_element)
        frame_element = show_node.process(frame_element)
        if save_video:
            video_saver_node.process(frame_element)

        if isinstance(frame_element, VideoEndBreakElement):
            break

        if show_in_web:
            video_server_node.update_image(frame_element.frame_result)

        if PRINT_PROFILE_INFO:
            print(
                f"PROC_PROCESSOR: {(time()-ts0) * 1000:.0f} ms: "
                + f"get {(ts1-ts0) * 1000:.0f} | "
                + f"nodes_inference {(time()-ts1) * 1000:.0f} | "
            )


@hydra.main(version_base=None, config_path="configs", config_name="app_config")
def main(config) -> None:
    time_sleep_start = 5

    frame_queue_max_size = 2
    frame_queue = Queue(frame_queue_max_size)

    frame_process = Process(target=proc_frame_reader, args=(frame_queue, config, time_sleep_start))
    frame_process.daemon = True
    frame_process.start()

    proc_proceessor(frame_queue, config, frame_process)


if __name__ == "__main__":
    ts = time()
    main()
    print(f"\n total time: {(time()-ts) / 60:.2} minute")
