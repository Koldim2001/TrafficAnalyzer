from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement
from utils_local.utils import profile_time
import psycopg2


class SentInfoDBNode:
    """Модуль для отправки актуальной информации о трафике в базу данных"""

    def __init__(self, config: dict) -> None:
        config_db = config["sent_info_db_node"]
        self.how_often_add_info = config_db["how_often_add_info"]
        self.table_name = config_db["table_name"]

        # Параметры подключения к базе данных
        db_connection = config["connection_info"]
        conn_params = {
            "user": db_connection["user"],
            "password": db_connection["password"],
            "host": db_connection["host"],
            "port": str(db_connection["5488"]),
            "database": db_connection["database"],
        }

        self.buffer_analytics_sec = (
            config["general"]["buffer_analytics"] * 60 +
            config["general"]["min_time_life_track"]
        )  # столько по времени буфер набирается и информацию о статистеке выводить рано

        # Подключение к базе данных
        try:
            self.connection = psycopg2.connect(**conn_params)
            print("Connected to PostgreSQL")
        except (Exception, psycopg2.Error) as error:
            print("Error while connecting to PostgreSQL:", error)

        # Создание курсора для выполнения SQL-запросов
        self.cursor = self.connection.cursor()

        # SQL-запрос для удаления таблицы, если она уже существует
        drop_table_query = f"DROP TABLE IF EXISTS {self.table_name};"

        # Удаление таблицы, если она уже существует
        try:
            self.cursor.execute(drop_table_query)
            self.connection.commit()
        except (Exception, psycopg2.Error) as error:
            print("Error while dropping table:", error)

        # SQL-запрос для создания таблицы
        create_table_query = f"""
        CREATE TABLE {self.table_name} (
            id SERIAL PRIMARY KEY,
            timestamp INTEGER,
            timestamp_date TIMESTAMP,
            cars INTEGER,
            road_1 FLOAT,
            road_2 FLOAT,
            road_3 FLOAT,
            road_4 FLOAT,
            road_5 FLOAT,
        );
        """

        # Создание таблицы
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            print("Table created successfully")
        except (Exception, psycopg2.Error) as error:
            print("Error while creating table:", error)


            
    @profile_time 
    def process(self, frame_element: FrameElement) -> FrameElement:
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element
        assert isinstance(
            frame_element, FrameElement
        ), f"CalcStatisticsNode | Неправильный формат входного элемента {type(frame_element)}"

        # Получение значений для записи в бд новой строки:
        info_dictionary = frame_element.info
        timestamp = frame_element.timestamp
        timestamp_date = frame_element.timestamp_date

        # для дебага оставил:
        if frame_element.frame_num % 100 == 0:
            print(info_dictionary)
            
            frame_element.send_info_of_frame_to_db = True

        return frame_element
