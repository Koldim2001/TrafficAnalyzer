FROM python:3.10.13

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --upgrade pip
RUN pip3 install "numpy<2"
RUN pip3 install cython_bbox==0.1.5 lap==0.4.0 
RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Сначала копируем только requirements.txt и устанавливаем зависимости
COPY requirements.txt /app/
RUN pip3 install -r requirements.txt

# Затем копируем остальной код
COPY . /app
