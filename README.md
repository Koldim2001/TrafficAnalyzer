# Анализ трафика на круговом движении

Данная программа выполняет анализ входящего трафика на участке кругового движения. Алгоритм определяет загруженность примыкающих дорог и выводит интерактивную статистику.

## Установка:
Необходима версия Python >= 3.10 (лучше 3.10.13)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
## Работа с программой:
Перед запуском необходимо в файле __configs/app_config.yaml__ указать все желаемые параметры. Далее можно запускать код.

Классический запуск кода:
```
python main.py
```
Пример запуска в дебаг режиме (профилировщик):
```
python main.py hydra.job_logging.root.level=DEBUG
```

Можно гиперпараметры из конфига configs/app_config.yaml для конкретного запуска поменять прямо в коде:
```
python main.py --config-name=app_config
```
---

__Пример работы алгоритма__: каждая машина отображается цветом, соответствующим дороге, с которой она прибыла к круговому движению:

![Traffic Analysis](content_for_readme/road_analytics.gif)

---
Пример трекинга машин (каждый id своим уникальным цветом отображается) <br/>
Отображается таким образом при выборе в конфигурации show_node.show_track_id_different_colors=True 

![Traffic Tracking](content_for_readme/traffic_tracking.gif)

