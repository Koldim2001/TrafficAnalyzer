import hydra
from nodes.VideoReader import VideoReader
from nodes.ShowNode import ShowNode
from nodes.VideoSaverNode import VideoSaverNode
from nodes.DetectionTrackingNodes import DetectionTrackingNodes
from nodes.TrackerInfoUpdateNode import TrackerInfoUpdateNode
from nodes.CalcStatisticsNode import CalcStatisticsNode
from nodes.SendInfoDBNode import SendInfoDBNode


@hydra.main(version_base=None, config_path="configs", config_name="app_config")
def main(config) -> None:
    video_reader = VideoReader(config["video_reader"])
    detection_node = DetectionTrackingNodes(config)
    tracker_info_update_node = TrackerInfoUpdateNode(config)
    calc_statistics_node = CalcStatisticsNode(config)
    show_node = ShowNode(config)
    video_saver_node = VideoSaverNode(config["video_saver_node"])

    save_video = config["pipeline"]["save_video"]
    send_info_db = config["pipeline"]["send_info_db"]

    if send_info_db:
        send_info_db_node = SendInfoDBNode(config)

    for frame_element in video_reader.process():

        frame_element = detection_node.process(frame_element)
        frame_element = tracker_info_update_node.process(frame_element)
        frame_element = calc_statistics_node.process(frame_element)

        if send_info_db:
            frame_element = send_info_db_node.process(frame_element)

        frame_element = show_node.process(frame_element)

        if save_video:
            video_saver_node.process(frame_element)


if __name__ == "__main__":
    main()
