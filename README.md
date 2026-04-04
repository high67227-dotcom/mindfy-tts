# 🎙️ Qwen3-TTS

A powerful Text-to-Speech application featuring voice cloning, voice design, and custom voice generation powered by Alibaba's Qwen3-TTS models.

![Qwen3-TTS Demo](https://img.shields.io/badge/Gradio-UI-orange) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![CUDA](https://img.shields.io/badge/CUDA-12.8-green)

## ✨ Features

- **🎨 Voice Design** - Create custom voices using natural language descriptions
- **🎭 Voice Clone** - Clone any voice from a reference audio sample
- **🗣️ Custom Voice** - Generate speech with predefined speakers and style instructions
- **📝 Long Text Chunking** - Automatically splits long text at sentence/word boundaries
- **🎤 Whisper Transcription** - Auto-transcribe reference audio for voice cloning
- **⚙️ Model Management** - Download, load, and unload models with a simple UI

## 🖥️ Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.8 support
- ~8GB+ VRAM for 0.6B models, ~16GB+ for 1.7B models

## 📦 Installation (Windows)

### 1. Clone the repository

```bash
git clone https://github.com/SUP3RMASS1VE/Qwen3-TTS.git
cd Qwen3-TTS
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install uv (fast package installer)

```bash
pip install uv
```

### 4. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 5. Install PyTorch with CUDA 12.8

```bash
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

### 6. Install Triton for Windows

```bash
uv pip install triton-windows==3.3.1.post19
```

### 7. Install Flash Attention (optional but recommended)

```bash
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.10/flash_attn-2.8.2+cu128torch2.7-cp310-cp310-win_amd64.whl
```

## 🚀 Usage

### Start the application

```bash
python app.py
```

The Gradio UI will launch at `http://localhost:7860`

### Available Models

| Model Type | Sizes | Description |
|------------|-------|-------------|
| VoiceDesign | 1.7B | Create voices from text descriptions |
| Base | 0.6B, 1.7B | Voice cloning from reference audio |
| CustomVoice | 0.6B, 1.7B | TTS with predefined speakers |

### Tips

- Models auto-download on first use, or pre-download them in the **Models** tab
- Use the **Chunk Size** slider for long texts (splits at sentence boundaries)
- Click **🔄 Refresh Status** to see models loaded from other tabs
- **Whisper** auto-unloads after transcription to free GPU memory

## 📁 Project Structure

```
Qwen3-TTS/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── qwen_tts/             # Core TTS module
    ├── core/             # Model configurations and implementations
    │   ├── models/       # Qwen3-TTS model code
    │   ├── tokenizer_12hz/
    │   └── tokenizer_25hz/
    └── inference/        # High-level inference wrappers
```

## 🎛️ Supported Languages

Auto, Chinese, English, Japanese, Korean, French, German, Spanish, Portuguese, Russian

## 🎭 Available Speakers (CustomVoice)

Aiden, Dylan, Eric, Ono_anna, Ryan, Serena, Sohee, Uncle_fu, Vivian

## 🙏 Credits

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team
- [OpenAI Whisper](https://github.com/openai/whisper) for transcription
- [Gradio](https://gradio.app/) for the web UI

## 📄 License

This project uses models from Alibaba's Qwen team. Please refer to the original [Qwen3-TTS repository](https://github.com/QwenLM/Qwen3-TTS) for model licensing information.

