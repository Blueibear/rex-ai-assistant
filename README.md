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
# On Windows PowerShell:
.\.venv\Scriptsctivate
# On macOS / Linux:
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-ml.txt  # optional
pip install -r requirements-dev.txt  # optional
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
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

### Select Audio Devices (Optional)

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

Say your wake word and interact. To exit: Ctrl+C or type "exit".

---

## 🔉 TTS HTTP API

Start the API server:

```bash
python rex_speak_api.py
```

Send a request:

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

| Variable | Purpose | Notes / Default |
|---|---------|-----------------|
| `REX_ACTIVE_USER` | Select initial user profile | defaults to first profile |
| `REX_WAKEWORD` | Desired wake phrase | fallback ONNX if no custom model |
| `REX_WAKEWORD_THRESHOLD` | Sensitivity level | default ~0.5 |
| `WHISPER_MODEL` / `REX_WHISPER_MODEL` | Whisper model size | e.g. `tiny`, `base`, `small`, `medium`, `large` |
| `WHISPER_DEVICE` / `REX_WHISPER_DEVICE` | “cpu” or “cuda” | `cuda` recommended if GPU available |
| `REX_LLM_MODEL`, `REX_LLM_MAX_TOKENS`, `REX_LLM_TEMPERATURE` | Transformer settings | see defaults |
| `SERPAPI_KEY`, `SERPAPI_ENGINE` | Web search via SerpAPI | if none, uses DuckDuckGo fallback |
| `REX_SPEAK_API_KEY` | API key for TTS endpoint | optional, for security |
| `REX_PROXY_TOKEN`, `REX_PROXY_ALLOW_LOCAL` | For Flask proxy or Cloudflare Access use | – |

---

## 🧠 Memory & Profiles

Under `Memory/<user>/` you’ll find:

- `core.json` — metadata and default settings  
- `history.jsonl` — chronologically appended chat entries  
- (Optional) voice sample file, notes, and others  

Profiles allow Rex to remember preferences, vocabulary, and voice traits.

---

## 🛠️ Optional Tools

- `record_wakeword.py` — record or train a custom ONNX wake-word model  
- `wakeword_listener.py` — script that just listens and beeps on wake detection  
- `flask_proxy.py` — reverse-proxy wrapper (for use with Cloudflare Access)  
- `rex_speak_api.py` — TTS-only HTTP interface  
- `plugin` folder — place plugins (e.g. `web_search`) and they will auto-load  

---

## 🧪 Tests & CI

```bash
pytest
```

CI is handled by GitHub Actions via `.github/workflows/ci.yml`.

---

## 🐳 Docker Support (Optional)

If using Docker, your image should:

- Use a CUDA-enabled base image (e.g. `nvidia/cuda:11.8-runtime`)
- Install `ffmpeg`, `libsndfile`, `portaudio`, and `nvidia-cuda-toolkit`
- Use `--index-url https://download.pytorch.org/whl/cu118` to install PyTorch CUDA wheels

---

## 📄 License

MIT License — feedback and contributions welcome!