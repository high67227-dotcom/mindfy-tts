# Берем свежую мощную коробку: PyTorch 2.2.0 и CUDA 12.1
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Устанавливаем системный пакет ffmpeg для работы с аудио
RUN apt-get update && apt-get install -y ffmpeg

# Копируем код
COPY . /app

# Устанавливаем зависимости
RUN pip install -r requirements.txt
# ДОБАВЛЯЕМ librosa и soundfile для чтения твоих mp3-голосов
RUN pip install runpod scipy librosa soundfile

# Запускаем ТОЛЬКО наш новый обработчик
CMD python handler.py
