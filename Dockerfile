# Берем более свежую коробку: PyTorch 2.2.0 и CUDA 12.1
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Устанавливаем системный пакет ffmpeg, а также SOX для работы с аудио
RUN apt-get update && apt-get install -y \
    ffmpeg \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Копируем код нейросети
COPY . /app

# Устанавливаем Python библиотеки
RUN pip install -r requirements.txt
RUN pip install runpod gradio_client

# Устанавливаем Flash Attention (благодаря devel-образу скомпилируется без проблем)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Запускаем нейросеть и обработчик
CMD python app.py & python handler.py
