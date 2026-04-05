# Берем свежую мощную коробку: PyTorch 2.2.0 и CUDA 12.1
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Устанавливаем системный пакет ffmpeg для работы с аудио
RUN apt-get update && apt-get install -y ffmpeg

# Копируем код
COPY . /app

# Устанавливаем зависимости (gradio_client нам больше не нужен)
RUN pip install -r requirements.txt
RUN pip install runpod scipy

# Запускаем ТОЛЬКО наш новый обработчик (без app.py)
CMD python handler.py
