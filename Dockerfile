# Берем более свежую коробку: PyTorch 2.2.0 и CUDA 12.1
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Устанавливаем системный пакет ffmpeg для работы с аудио
RUN apt-get update && apt-get install -y ffmpeg

# Копируем код нейросети
COPY . /app

# Устанавливаем Python библиотеки
RUN pip install -r requirements.txt
RUN pip install runpod gradio_client

# Запускаем нейросеть и обработчик
CMD python app.py & python handler.py
