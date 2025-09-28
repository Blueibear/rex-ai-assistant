# 🧠 Rex AI Assistant

Rex is a **local-first**, voice-driven AI assistant. It supports a full pipeline:

- Wake-word detection  
- Speech-to-text via Whisper  
- Transformer-based response generation  
- Text-to-speech via Coqui XTTS (voice cloning if a sample is provided)  
- Optional web search plugin  
- HTTP TTS API endpoint  
- Memory / personalization via per-user profiles  

All audio processing, inference, and data stay on your machine unless you explicitly provide cloud keys.

---

## ✨ Features & Architecture

- 🔊 **Wake-word detection** with openWakeWord, supporting custom ONNX models  
- 🗣️ **Speech-to-text** using OpenAI’s Whisper model  
- 🤖 **Transformer inference** via Hugging Face (or registerable backends)  
- 🔉 **Text-to-speech** via Coqui XTTS, with optional user voice cloning  
- 🌐 **Web search plugin** — attempts SerpAPI first, falls back to DuckDuckGo scraping  
- 🛡 **Flask TTS API** — a minimal HTTP interface for TTS with API key / proxy support  
- 🧠 **Memory / user profiles** — structured metadata, history, notes, voice sample references  
- ✅ **Unit tests & CI** to guard regressions  

---

## 📦 Quick Start

### Prerequisites

- Python **3.10+**  
- Git  
- FFmpeg installed and in PATH (for audio encoding/decoding)  
- A microphone & speaker  
- NVIDIA RTX GPU (optional, for accelerating Whisper / Transformers / TTS)  

---

### Setup

```bash
git clone https://github.com/Blueibear/rex-ai-assistant.git
cd rex-ai-assistant

python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install --upgrade pip

# Install CPU fallback of PyTorch first (helps avoid CUDA issues if not set up)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Core runtime (wake word, Flask API, etc.)
pip install -r requirements.txt

# Optional: install speech + language model stack (Whisper, Torch, XTTS)
pip install -r requirements-ml.txt

# Optional: developer tooling (pytest, coverage)
pip install -r requirements-dev.txt

# Or run:
# python install.py --with-ml --with-dev
```

---

### Enable GPU (CUDA) Support

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Verify:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

### Select Audio Devices

```bash
python audio_devices.py --list
python audio_devices.py --set-input <device_id>
python audio_devices.py --set-output <device_id>
```

---

### Launch the Assistant

```bash
python rex_loop.py
```

---

### TTS HTTP API

```bash
python rex_speak_api.py
```

Send:

```http
POST /speak
Content-Type: application/json
X-API-Key: your_secret

{
  "text": "Hello Rex",
  "user": "james"
}
```

Returns a WAV file.

---

## ⚙️ Configuration & Environment Variables

See `.env.example` for all config options:

| Variable | Description |
|---|---|
| REX_ACTIVE_USER | Profile name |
| REX_WAKEWORD | Wake phrase |
| WHISPER_MODEL | Whisper size (`base`, `medium`, etc.) |
| WHISPER_DEVICE | `cpu` or `cuda` |
| REX_LLM_MODEL | Language model backend |
| SERPAPI_KEY | For web search |
| REX_SPEAK_API_KEY | TTS API auth key |

---

## 🧠 Memory & Profiles

- `Memory/<user>/core.json`: user settings  
- `Memory/<user>/history.jsonl`: chat history  
- (Optional) voice sample, notes, etc.

---

## 🛠️ Tools

- `record_wakeword.py`: collect training samples  
- `wakeword_listener.py`: debug wakeword  
- `rex_speak_api.py`: run HTTP server  
- `flask_proxy.py`: optional reverse proxy  
- `plugin/`: drop-in plugins like web search  

---

## 🧪 Tests

```bash
pytest
```

---

## 🐳 Docker (Optional)

Use a base like:

```
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04
```

And install:

- ffmpeg
- libsndfile1
- portaudio19-dev
- nvidia-cuda-toolkit
- CUDA-enabled PyTorch wheels via `--index-url`

---

## 📄 License

MIT License.

