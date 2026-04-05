import runpod
import base64
import io
import torch
import numpy as np
import scipy.io.wavfile as wavfile
from qwen_tts import Qwen3TTSModel # Импортируем движок напрямую!
import re

# 1. ЗАГРУЖАЕМ МОДЕЛЬ ОДИН РАЗ ПРИ СТАРТЕ СЕРВЕРА
print("Загрузка модели Qwen3-TTS в память...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", # Берем CustomVoice, так как он быстрый
    device_map="cuda",
    dtype=torch.bfloat16,
)
print("Модель загружена успешно!")

def chunk_text(text: str) -> list:
    """Бьем текст на предложения прям внутри видеокарты"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]

# 2. ФУНКЦИЯ ОБРАБОТКИ ПОТОКА
def handler(job):
    job_input = job['input']
    text = job_input.get('text', '')
    speaker = job_input.get('speaker', 'johndoe')
    
    if not text:
        return {"error": "Пустой текст"}

    chunks = chunk_text(text)
    
    # 3. ГЕНЕРИРУЕМ И СРАЗУ ОТДАЕМ (YIELD)
    for i, chunk in enumerate(chunks):
        try:
            # Прямой вызов нейросети (без Gradio!)
            wav_data, sr = model.generate_custom_voice(
                text=chunk,
                language="Auto",
                speaker=speaker,
                non_streaming_mode=True
            )
            
            # Запаковываем сырые данные в WAV-байты в оперативной памяти (без диска)
            byte_io = io.BytesIO()
            wavfile.write(byte_io, sr, wav_data[0])
            audio_bytes = byte_io.getvalue()
            
            # Кодируем в Base64 для безопасной передачи
            chunk_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # МАГИЯ RUNPOD: Отдаем готовый кусок налету!
            yield {
                "chunk_index": i,
                "audio_base64": chunk_base64,
                "is_final": i == (len(chunks) - 1)
            }
            
        except Exception as e:
            yield {"error": str(e)}

# Запускаем Serverless с поддержкой генераторов (return_aggregate_stream)
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True # Это включает режим стриминга!
})
