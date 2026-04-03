# AskRex Assistant — Windows Quickstart

This guide covers Windows-specific setup for AskRex Assistant. For full project details, see [README.md](README.md).

---

## Quick Setup on Windows

### 1. Prerequisites

- Python **3.11**
- Git
- [FFmpeg](https://ffmpeg.org/download.html) — must be on PATH
- Microphone + speakers (for voice mode)
- NVIDIA GPU with CUDA 11.8+ (optional — CPU-only install also works)

### 2. Clone and create a virtual environment

```powershell
git clone https://github.com/Blueibear/AskRex-Assistant.git
cd rex-ai-assistant

py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

**CPU only:**
```powershell
pip install --upgrade pip setuptools wheel
pip install .
```

**GPU (CUDA 12.4 — RTX 30xx / 40xx):**
```powershell
pip install --upgrade pip setuptools wheel
pip install -r requirements-gpu-cu124.txt
```

Verify CUDA after GPU install:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 4. Configure Rex

Copy the example config files and fill in your values:

```powershell
copy config\rex_config.example.json config\rex_config.json
copy .env.example .env
```

- **`config\rex_config.json`** — runtime settings (model, audio device, wake word, feature flags).
  Edit this file to configure Rex behaviour.
- **`.env`** — secrets only (API keys, tokens). See the inline comments in `.env.example`.

---

## Runtime Modes

Rex has four distinct startup modes. Run the one that fits your use case:

### Text chat

Interactive text conversation with no audio hardware required:

```powershell
python -m rex
```

### Voice loop

Full wake word → speech-to-text → LLM → text-to-speech pipeline:

```powershell
python rex_loop.py
```

Say the wake word (default: **rex**) to activate voice input.

### Dashboard (GUI)

Desktop GUI for configuration, conversation history, and status:

```powershell
python run_gui.py
```

### TTS API

Local Flask API that accepts text and returns synthesised audio over HTTP:

```powershell
python rex_speak_api.py
```

The API binds to `http://localhost:5001` by default and requires an API key configured in `.env`.

---

## Run tests

```powershell
pytest -q
```

---

## Health check

```powershell
python -m rex doctor
```

---

## Notes

- Memory profiles live under `Memory/<user>/`
- Voices can be customised by adding a WAV file to your profile
- Web search requires a `SERPAPI_KEY` or `BRAVE_API_KEY` in `.env`; DuckDuckGo scraping is used as a fallback
- See [CONFIGURATION.md](CONFIGURATION.md) for the full configuration reference
- The full Windows voice stack is currently validated on Python 3.11 only. Python 3.13/3.14 installs are known to fail in the TTS/ML dependency path, so the installer now rejects them immediately.
