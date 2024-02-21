# Анализ трафика на круговом движении
### Установка:
Необходима версия Python >= 3.10 (лучше 3.10.13)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
### Работа с программой:
Перед запуском необходимо в файле configs/app_config.yaml указать все желаемые параметры. Далее можно запускать код.

Классический запуск кода:
```
python main.py
```
Пример запуска в дебаг режиме:
```
python main.py hydra.job_logging.root.level=DEBUG
```

Можно гиперпараметры из конфига configs/app_config.yaml для конкретного запуска поменять прямо в коде:
```
python main.py --config-name=app_config
```
---
Пример трекинга машин:

![Traffic Tracking](content_for_readme/traffic_tracking.gif)

