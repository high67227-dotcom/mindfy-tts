# Берем готовую операционную систему с установленным PyTorch и CUDA для видеокарт
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Копируем код нейросети внутрь контейнера
COPY . /app

# Устанавливаем библиотеки
RUN pip install -r requirements.txt
RUN pip install runpod gradio_client

# При старте контейнера запускаем нейросеть в фоне (&) и наш обработчик
CMD python app.py & python handler.py