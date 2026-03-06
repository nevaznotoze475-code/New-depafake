#!/bin/bash

echo "🚀 Установка Deepfake бота..."

# Обновление системы
apt-get update
apt-get upgrade -y

# Установка системных пакетов
apt-get install -y python3-pip python3-dev git ffmpeg \
    libgl1-mesa-glx libglib2.0-0 wget

# Установка CUDA (для GPU)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get -y install cuda-12-1

# Установка Python пакетов
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Скачивание моделей
python3 download_models.py

echo "✅ Установка завершена!"
echo "👉 Отредактируйте .env файл и запустите: python3 bot.py"
