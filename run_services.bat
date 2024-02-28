REM Создаем необходимые папки
mkdir .\services\pg_data_wh
mkdir .\services\pg_grafana
mkdir .\services\grafana

REM Запускаем docker-compose
docker-compose -p traffic_analyzer up -d