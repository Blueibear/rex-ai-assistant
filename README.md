# Rex AI Assistant

> A local-first, voice-activated AI companion that runs entirely on your machine with wake word detection, speech recognition, LLM chat, and text-to-speech.

<p align="center">
  <img src="https://github.com/Blueibear/rex-ai-assistant/actions/workflows/ci.yml/badge.svg" alt="CI status" />
</p>

## Features

- 🔊 **Wake word detection** via openWakeWord (customizable trigger phrases)
- 🗣️ **Speech-to-text** using OpenAI Whisper (runs offline)
- 🤖 **LLM responses** via Transformers (local), OpenAI API, or Ollama
- 🔉 **Text-to-speech** with Coqui XTTS, edge-tts, or pyttsx3 (voice cloning supported)
- 🌐 **Web search plugins** for SerpAPI, Brave, Google CSE, and DuckDuckGo
- 🧠 **Per-user memory** profiles with conversation history and preferences
- 🔐 **Flask TTS API** with authentication and rate limiting
- ✅ **CI/CD** with GitHub Actions and Release Please automation
- 🐳 **Docker support** for containerized deployment

## Requirements

| Component     | Requirement                            |
|--------------|-----------------------------------------|
| **OS**       | macOS 11+, Windows 10+, or Ubuntu 20.04+ |
| **Python**   | 3.9 or newer (3.10+ recommended)        |
| **FFmpeg**   | Must be installed and available on PATH |
| **Hardware** | Microphone and speakers for voice mode  |
| **GPU** (optional) | NVIDIA GPU with CUDA 11.8+        |

## Quickstart

🔉 Text-to-speech via Coqui XTTS, with voice cloning support

### macOS/Linux (bash)

```bash
# Clone repository
git clone https://github.com/Blueibear/rex-ai-assistant.git
cd rex-ai-assistant

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Copy environment template
cp .env.example .env
# Edit .env with your preferred editor to set API keys and options

# Install dependencies (CPU-only PyTorch)
pip install --upgrade pip setuptools wheel
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cpu
pip install -e .

# Run health check
python scripts/doctor.py

# Start text-based chat mode
python -m rex

# Or start full voice assistant with wake word
python rex_loop.py
