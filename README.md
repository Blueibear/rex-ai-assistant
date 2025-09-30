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

# Create and activate a virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scriptsctivate
# On macOS / Linux:
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Optional: install speech + language model stack (Whisper, Torch, XTTS)
pip install -r requirements-ml.txt

# Optional: developer tooling (pytest, coverage)
pip install -r requirements-dev.txt
```

---

### Enable GPU (CUDA) Support

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Verify with:

```python
import torch
print(torch.__version__)               
print(torch.cuda.is_available())       
print(torch.cuda.get_device_name(0))   
```

---

### Audio Device Setup

```bash
python audio_devices.py --list
python audio_devices.py --set-input <device_id>
python audio_devices.py --set-output <device_id>
```

---

### Run the Assistant

```bash
python rex_loop.py
```

---

### TTS HTTP API

```bash
python rex_speak_api.py
```

POST to `/speak`:

```http
POST /speak
Content-Type: application/json
X-API-Key: your_secret

{
  "text": "Hello Rex",
  "user": "james"
}
```

---

## ⚙️ Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `REX_ACTIVE_USER` | Active profile | auto-detected |
| `REX_WAKEWORD` | Wake phrase | `rex` |
| `REX_WAKEWORD_THRESHOLD` | Sensitivity | `0.5` |
| `WHISPER_MODEL` | Whisper size | `medium` |
| `WHISPER_DEVICE` | CPU/GPU | `cuda` |
| `REX_LLM_MODEL`, `REX_LLM_MAX_TOKENS`, `REX_LLM_TEMPERATURE` | Language model settings | – |
| `SERPAPI_KEY` | Enables SerpAPI search | optional |
| `REX_SPEAK_API_KEY` | Secures `/speak` endpoint | optional |
| `REX_PROXY_TOKEN`, `REX_PROXY_ALLOW_LOCAL` | Proxy config | – |

---

## 🧠 Memory / Profiles

Each user folder under `Memory/` includes:

- `core.json` — metadata  
- `history.jsonl` — chat logs  
- voice samples and notes

---

## 🧪 Tests & CI

Run tests:

```bash
pytest
```

GitHub Actions will run tests via `.github/workflows/ci.yml`.

---

## 🐳 Docker Notes

If using Docker:

- Use `nvidia/cuda:11.8-runtime` base image  
- Install `ffmpeg`, `portaudio`, `libsndfile`  
- Install PyTorch CUDA wheels via PyPI index URL  

---

## ℹ️ Troubleshooting

- **CUDA issues** → verify drivers and device availability  
- **Mic/speaker errors** → use `audio_devices.py` to inspect  
- **Plugins missing** → check plugin structure and logs  

---

## 📄 License

MIT License  
Feedback and contributions welcome!
