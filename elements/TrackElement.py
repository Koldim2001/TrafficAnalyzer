
class TrackElement:
    # Класс, содержаций информацию о конкретном треке машины
    def __init__(
        self,
        id: int,
        timestamp_first: float,
        timestamp_last: float | None = None,
        start_road: int | None = None,
    ) -> None:
        self.id = id  
        self.timestamp_first = timestamp_first  
        self.timestamp_last = timestamp_last
        self.start_road = start_road    