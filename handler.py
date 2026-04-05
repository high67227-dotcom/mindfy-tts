import time
import runpod
import base64
import tempfile
import os
from gradio_client import Client, handle_file

print("⏳ Ожидание запуска нейросети Qwen3-TTS...")
client = None
# Ждем, пока локальный интерфейс нейросети запустится (до 60 секунд)
for i in range(30):
    try:
        client = Client("http://127.0.0.1:7860")
        print("✅ Нейросеть готова к работе!")
        break
    except Exception:
        time.sleep(2)

if not client:
    raise Exception("❌ Не удалось запустить локальный Gradio")

def generate_audio(job):
    job_input = job["input"]
    text = job_input.get("text")
    ref_audio_b64 = job_input.get("ref_audio_base64")
    ref_text = job_input.get("ref_text", "")

    # Распаковываем аудио из Base64 во временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_ref:
        temp_ref.write(base64.b64decode(ref_audio_b64))
        temp_ref_path = temp_ref.name

    try:
        # Отправляем запрос в локальную нейросеть
        result = client.predict(
            ref_audio=handle_file(temp_ref_path),
            ref_text=ref_text,
            target_text=text,
            language="Auto",
            use_xvector_only=True,
            model_size="1.7B",
            max_chunk_chars=200,
            chunk_gap=0.0,
            seed=-1,
            api_name="/generate_voice_clone",
        )
        
        raw_path = result[0] if isinstance(result, (list, tuple)) else result
        status_msg = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else "Неизвестно"
        
        if raw_path is None:
            raise Exception(f"ОШИБКА ОТ НЕЙРОСЕТИ: {status_msg}")
        
        # Упаковываем готовое аудио обратно в Base64
        with open(raw_path, "rb") as f:
            output_base64 = base64.b64encode(f.read()).decode('utf-8')
            
        return {"audio_base64": output_base64}
        
    finally:
        os.remove(temp_ref_path)

# Запускаем прослушивание Serverless запросов
runpod.serverless.start({"handler": generate_audio})
