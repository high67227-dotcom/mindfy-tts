import runpod
import base64
import io
import torch
import numpy as np
import scipy.io.wavfile as wavfile
from qwen_tts import Qwen3TTSModel
import re
import tempfile
import os
import librosa

print("Загрузка модели Qwen3-TTS Base (Voice Clone) в память...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base", # Возвращаем версию Base для клонирования
    device_map="cuda",
    dtype=torch.bfloat16,
)
print("Модель загружена!")

def chunk_text(text: str) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]

def decode_audio(b64_str):
    # Временно сохраняем и читаем аудио
    audio_bytes = base64.b64decode(b64_str)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
        
    wav, sr = librosa.load(tmp_path, sr=None)
    os.unlink(tmp_path)
    return wav, sr

def handler(job):
    job_input = job['input']
    text = job_input.get('text', '')
    ref_audio_base64 = job_input.get('ref_audio_base64', '')
    ref_text = job_input.get('ref_text', '')
    
    if not text or not ref_audio_base64:
        yield {"error": "Нет текста или референсного аудио"}
        return

    try:
        ref_wav, ref_sr = decode_audio(ref_audio_base64)
        ref_audio_tuple = (ref_sr, ref_wav)
    except Exception as e:
        yield {"error": f"Ошибка аудио: {e}"}
        return

    chunks = chunk_text(text)
    
    for i, chunk in enumerate(chunks):
        try:
            # ИСПРАВЛЕНО: Убрали non_streaming_mode, добавили max_new_tokens
            wav_data, sr = model.generate_voice_clone(
                text=chunk,
                language="Auto",
                ref_audio=ref_audio_tuple,
                ref_text=ref_text,
                x_vector_only_mode=False,
                max_new_tokens=2048
            )
            
            byte_io = io.BytesIO()
            wavfile.write(byte_io, sr, wav_data[0])
            audio_bytes = byte_io.getvalue()
            chunk_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            yield {
                "chunk_index": i,
                "audio_base64": chunk_base64,
                "is_final": i == (len(chunks) - 1)
            }
        except Exception as e:
            # ИСПРАВЛЕНО: Теперь ошибка будет громко выводиться в логи RunPod
            print(f"🔥 ОШИБКА ГЕНЕРАЦИИ ВНУТРИ RUNPOD: {str(e)}")
            yield {"error": str(e)}

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True
})
