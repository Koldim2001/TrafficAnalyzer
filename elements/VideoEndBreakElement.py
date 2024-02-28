from elements.FrameElement import FrameElement


class VideoEndBreakElement(FrameElement):
    """Элемемент прерывания видеопотока"""

    def __init__(self, video_source, timestamp) -> None:
        """ "Элемент конца видеопотока. Используется для возможности остановить
        все активные процессы в main_optimized.py по окончанию кадров.

        Args:
            video_source (_type_): video_source (str): guid источника изображения
                (номер камеры, если читаем стрим, иначе - название файла);
            timestamp (_type_): время считывания изображения в секундах.
        """
        self.video_source = video_source
        self.timestamp = timestamp
