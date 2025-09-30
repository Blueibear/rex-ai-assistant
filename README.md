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
source .venv/bin/activate

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Optional: install speech + language model stack (Whisper, Torch, XTTS)
pip install -r requirements-ml.txt

# Optional: developer tooling (pytest, coverage)
pip install -r requirements-dev.txt

# Alternatively, you can run the installation helper:
# python install.py --with-ml --with-dev
```

---

### Enable GPU (CUDA) Support (for your RTX GPU)

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Verify:

```python
import torch
print(torch.__version__)               # should end in +cu118
print(torch.cuda.is_available())       # should be True
print(torch.cuda.get_device_name(0))   # should show your GPU
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

Say your wake word and interact.

---

### TTS HTTP API

```bash
python rex_speak_api.py
```

POST:

```json
{
  "text": "Hello Rex",
  "user": "james"
}
```

Returns a WAV file.

---

## ⚙️ Configuration & Environment Variables

| Variable | Description |
|---------|-------------|
| `REX_ACTIVE_USER` | Sets active memory/profile |
| `REX_WAKEWORD`, `REX_WAKEWORD_THRESHOLD` | Wake-word config |
| `WHISPER_MODEL`, `WHISPER_DEVICE` | Whisper model & device |
| `REX_LLM_MODEL`, `REX_LLM_MAX_TOKENS` | Language model settings |
| `REX_SPEAK_API_KEY` | Secures TTS endpoint |
| `SERPAPI_KEY` | Enables SerpAPI-based search |
| `REX_PROXY_TOKEN` | Token for reverse proxy |

---

## 🧠 Memory

Stored in `Memory/<user>/`:

- `core.json` — profile metadata
- `history.jsonl` — chronological interaction log

---

## 🧪 Tests

```bash
pytest
```

CI runs on GitHub Actions.

---

## 🐳 Docker

Use CUDA base image if GPU support is needed:
- Install ffmpeg, portaudio, nvidia-cuda-toolkit
- Install torch with CUDA index URL

---

## 📄 License

MIT License.