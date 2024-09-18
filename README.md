# Анализ трафика на круговом движении (prod версия с множеством камер)

Данная программа осуществляет анализ входящего трафика на участке кругового движения. Алгоритм определяет загруженность примыкающих дорог и выводит интерактивную статистику.

Подробный туториал по проекту - [__ссылка на видео__](https://youtu.be/u9EtqHz4Vqc)

## Установка:
```
docker-compose -p traffic_analyzer up -d --build
```

Необходимо в главной директории создать файл с переменными окружения, которые будут прокинуты во все контейнеры. Для этого создайте файл `secrets.txt` и положите подобный текст с паролями:
```
POSTGRES_DB=traffic_analyzer_db
POSTGRES_USER=user
POSTGRES_PASSWORD=pwd
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=admin
```
Каждая новая камера добавляется как +1 инстанс бекенда traffic_analyzer_camera_N в котором надо указать лишь разные scr и конфигурации через переменные окружения

У каждой камеры сой дашборд между которыми можно переходит по кнопке:

![grafana](https://github.com/user-attachments/assets/c0c6d602-2026-460f-9c48-64180e87ca8e)


 
## Работа с программой:
Перед запуском необходимо в файле __configs/app_config.yaml__ указать все желаемые параметры. Далее можно запускать код.

Чтобы запустить проект с определенным видео, необходимо указать путь к нему в докер компоузе переменной окружения

---
## Примеры работы кода:

__Пример работы алгоритма c выводом статистики__: каждая машина отображается цветом, соответствующим дороге, с которой она прибыла к круговому движению + выводится значение числа видимых машин + значения интенсивности входного потока (число машин в минуту с каждой входящей дороги). <br/>Отображается таким образом при выборе в конфигурации show_node.show_info_statistics=True 

![Traffic statistics 1](content_for_readme/with_statistics_1.gif)
![Traffic statistics 2](content_for_readme/with_statistics_2.gif)

Отключить отображение окна со статистикой можно при выборе в конфигурации show_node.show_info_statistics=False <br/>
Чтобы наблюдать fps обработки как в первом представленном примере, необходимо в конфиге указать show_node.draw_fps_info=True.  <br/>При наличии GPU получается достигнуть порядка 30-40 кадров в секунду в случае запуска __main_optimized.py__

---
__Пример режима демонстрации трекинга машин__ (каждый id своим уникальным цветом отображается) <br/>
Отображается таким образом при выборе в конфигурации show_node.show_track_id_different_colors=True 

![Traffic Tracking](content_for_readme/traffic_tracking.gif)

---
## Включение сторонних сервисов для визуализации результатов:
Программа позволяет вести запись актуальной статистики о машинопотоке в базу данных PostgreSQL и тут же осуществлять визуализацию в виде интерактивного дашборда Grafana.

![image](https://github.com/user-attachments/assets/90844b76-45d0-4223-822d-4e943138c338)

Тем самым у конечного потребителя этого приложения имеется возможность запустить код один раз, подключив на вход RTSP поток или заготовленный видеофайл, и постоянно получать актуальную статистику, а также просматривать историю загруженности участка движения.


## Вывод обработанного видеопотока в веб-интерфейс:

Обработанные кадры можно отображать в веб-интерфейсе (вместо отдельного окна OpenCV). Бэкенд сайта реализован с использованием Flask.

Для того, чтобы запустить проект таким образом, необходимо в файле configs/app_config.yaml в разделе pipeline указать show_in_web=True и в show_node указать imshow=False. Далее можно запускать main.py или main_optimized.py и переходить по ссылке http://localhost:8100/

Пример того, как можно запустить проект и иметь возможность одновременно смотреть стрим по порту 8100 и наблюдать интерактивный дашборд в Grafana по порту 3111:

![web+grafana](content_for_readme/web+grafana.gif)

---

## Рассмотрим, как реализован код:

Каждый кадр последовательно проходит через ноды, и в атрибуты этого объекта постепенно добавляется все больше и больше информации.

```mermaid
graph TD;
    A["VideoReader<br>Считывает кадры из видеофайла"] --> B["DetectionTrackingNodes<br>Реализует детектирование машин + трекинг"];
    B --> C["TrackerInfoUpdateNode<br>Обновляет информацию об актуальных треках"];
    C --> D["CalcStatisticsNode<br>Вычисляет загруженность дорог"];
    D --sent_info_db==False --> F;
    D --sent_info_db==True --> E["SentInfoDBNode<br>Отправляет результаты в базу данных"];
    E --> F["ShowNode<br>Отображает результаты на экране"];
    F --save_video==True --> H["VideoSaverNode<br>Сохраняет обработанные кадры"];
    F --show_in_web==True & save_video==False --> L["FlaskServerVideoNode<br>Обновляет кадры в веб-интерфейсе"];
    H --show_in_web==True --> L
```
