import hydra
from nodes.VideoReader import VideoReader
from nodes.ShowNode import ShowNode
from nodes.VideoSaverNode import VideoSaverNode
from nodes.DetectionNode import DetectionNode



@hydra.main(version_base=None, config_path="configs", config_name="app_config")
def main(config) -> None:
    video_reader = VideoReader(config["video_reader"])
    detection_node = DetectionNode(config["detection_node"])
    show_node = ShowNode(config)
    video_saver_node = VideoSaverNode(config["video_saver_node"])

    save_video = config["pipeline"]["save_video"]
    
    for frame_element in video_reader.process():
        
        frame_element = detection_node.process(frame_element)
        frame_element = show_node.process(frame_element)

        if save_video:
            video_saver_node.process(frame_element)


if __name__ == "__main__":
    main()
