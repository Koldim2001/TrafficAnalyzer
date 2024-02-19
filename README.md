# Анализ трафика на круговом движении
Установка:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
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
