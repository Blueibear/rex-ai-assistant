# ?? Rex AI Assistant

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

## ? Features & Architecture

- ?? **Wake-word detection** with openWakeWord, supporting custom ONNX models  
- ??? **Speech-to-text** using OpenAI’s Whisper model  
- ?? **Transformer inference** via Hugging Face (or registerable backends)  
- ?? **Text-to-speech** via Coqui XTTS, with optional user voice cloning  
- ?? **Web search plugin** — attempts SerpAPI first, falls back to DuckDuckGo scraping  
- ?? **Flask TTS API** — a minimal HTTP interface for TTS with API key / proxy support  
- ?? **Memory / user profiles** — structured metadata, history, notes, voice sample references  
- ? **Unit tests & CI** to guard regressions  

---

## ?? Quick Start

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
.\.venv\Scripts\activate
# On macOS / Linux:
# source .venv/bin/activate

pip install --upgrade pip
# Core runtime (wake word, Flask API, etc.)
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

Inside the same virtual environment:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Then verify:

```python
import torch
print(torch.__version__)               # should end in +cu118
print(torch.cuda.is_available())       # should be True
print(torch.cuda.get_device_name(0))   # should show your GPU
```

---

### Select Audio Devices (Optional)

If audio devices aren’t automatically detected or you want to change them:

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

Then speak your wake word and interact.  

To stop: type **exit**, **quit**, or press **Ctrl+C**.

---

### TTS HTTP API

Start the API server:

```bash
python rex_speak_api.py
```

Send a request:

```
POST /speak
Content-Type: application/json
X-API-Key: your_secret

{
  "text": "Hello Rex",
  "user": "james"
}
```

It returns a WAV file. Useful for non-voice clients.

---

## ?? Configuration & Environment Variables

Rex is configured via environment variables. Some important ones:

| Variable | Purpose | Notes / Default |
|---|---------|-----------------|
| `REX_ACTIVE_USER` | Select initial user profile | defaults to first profile |
| `REX_WAKEWORD` | Desired wake phrase | fallback ONNX if no custom model |
| `REX_WAKEWORD_KEYWORD` | Wake-word keyword fallback | defaults to `hey_jarvis` |
| `REX_WAKEWORD_THRESHOLD` | Sensitivity level | default ~0.5 |
| `WHISPER_MODEL` / `REX_WHISPER_MODEL` | Whisper model size | e.g. `tiny`, `base`, `small`, `medium`, `large` |
| `WHISPER_DEVICE` / `REX_WHISPER_DEVICE` | "cpu" or "cuda" | set to `cuda` when GPU available |
| `REX_LLM_MODEL`, `REX_LLM_MAX_TOKENS`, `REX_LLM_TEMPERATURE` | Transformer settings | see defaults |
| `REX_LLM_TOP_P`, `REX_LLM_TOP_K`, `REX_LLM_SEED` | Advanced sampling controls | optional overrides |
| `SERPAPI_KEY`, `SERPAPI_ENGINE` | Web search via SerpAPI | falls back to DuckDuckGo when missing |
| `REX_SPEAK_API_KEY` | API key for TTS endpoint | required for `rex_speak_api.py` |
| `REX_PROXY_TOKEN`, `REX_PROXY_ALLOW_LOCAL` | For Flask proxy or Cloudflare Access use | allow local dev when set |

---

## ?? Memory & Profiles

Under `Memory/<user>/` you’ll find:

- `core.json` — metadata and default settings  
- `history.jsonl` — chronologically appended chat entries  
- (Optional) voice sample file, notes, and others  

Profiles allow Rex to remember preferences, vocabulary, and voice traits.

---

## ??? Optional Tools

- `record_wakeword.py` — record or train a custom ONNX wake-word model  
- `wakeword_listener.py` — script that just listens and beeps on wake detection  
- `flask_proxy.py` — reverse-proxy wrapper (for use with Cloudflare Access)  
- `rex_speak_api.py` — TTS-only HTTP interface  
- `plugin` folder — place plugins (e.g. `web_search`) and they will auto-load  

---

## ?? Tests & CI

Run locally:

```bash
pytest
```

CI is set up via `.github/workflows/ci.yml` which runs on every `push` and `pull_request`. It installs system dependencies including `nvidia-cuda-toolkit` (for GPU support), then installs Python dependencies and runs tests with coverage.

---

## ?? (Optional) Docker Support / GPU Containers

If you containerize Rex, ensure your Dockerfile:

- Uses a CUDA-enabled base image (e.g. `nvidia/cuda:11.8-runtime`)
- Installs system dependencies: `ffmpeg`, `libsndfile`, `portaudio`, `nvidia-cuda-toolkit`
- Installs CUDA PyTorch wheels via the `--index-url https://download.pytorch.org/whl/cu118`

This ensures your container is GPU-ready.

---

## ?? Troubleshooting

- **CUDA not detected** ? check your GPU driver & CUDA installation  
- **Audio errors** ? run `python audio_devices.py --list` to check device indices  
- **Missing voice sample** ? voice cloning disabled, falls back to default  
- **Plugin errors** ? debug via logging; confirm plugin name in `plugins/`  

---

## ?? License & Acknowledgments

Rex is released under the **MIT License**.  
Contributions, feedback, and bug reports are welcome via GitHub.
