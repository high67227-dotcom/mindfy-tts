FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

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
