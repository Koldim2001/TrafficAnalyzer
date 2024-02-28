class TrackElement:
    # Класс, содержаций информацию о конкретном треке машины
    def __init__(
        self,
        id: int,
        timestamp_first: float,
        start_road: int | None = None,
    ) -> None:
        self.id = id  # Номер этого трека
        self.timestamp_first = timestamp_first  # Таймстемп инициализации (в сек)
        self.timestamp_last = timestamp_first  # Таймстемп последнего обнаружения (в сек)
        self.start_road = start_road  # Номер дороги, с которой приехал
        self.timestamp_init_road = timestamp_first  # Таймстемп инициализации номера дороги (в сек)
        # ps: если дорога не будет определена, то значение останется равным первому появлению

    def update(self, timestamp):
        # Обновление времени последнего обнаружения
        self.timestamp_last = timestamp
