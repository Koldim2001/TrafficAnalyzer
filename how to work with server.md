# Работа с сервером с GPU:

## Обновим пакеты и установим Git:
```
sudo apt update
sudo apt install git-all
```

---

## Установим драйвера GPU nvidia:
```
sudo apt-get install linux-headers-$(uname -r)
```
```
sudo apt-get install ubuntu-drivers-common
```
Можно глянуть какую версию ставить драйверов стоит ставить:
```
ubuntu-drivers devices | grep recommended
```
Установим драйвера и CUDA:
```
sudo ubuntu-drivers autoinstall
# либо указать напрямую: sudo apt install nvidia-driver-550 -y

sudo apt install nvidia-cuda-toolkit -y
```
Проверим поставилось ли все:
`nvidia-smi`

---

## Ставим Docker:

```
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

```
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
apt install docker-compose
```
Сделаем возможным работу с docker без sudo:
```
sudo groupadd docker
sudo usermod -aG docker $USER
```

---
## Docker NVIDIA Container Toolkit:
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```
Обновим конфиги и перезапустим докер:
```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
Для тестирования работоспособности ([доп ссылка](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.10.0/install-guide.html)):
```
docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```
---
## Portainer
Можно еще установить Portainer для того, чтобы у Docker был UI. Это позволяет удобнее взаимодействовать с докером. После поднятия сервиса он будет работать на порту 9000.

```
docker volume create portainer_data
docker run -d -p 8000:8000 -p 9443:9443 -p 9000:9000 --name portainer --restart=always -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer-ce:latest
```

---