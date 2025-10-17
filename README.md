<p align="center">
  <img src="assets/logo.svg" width="120" alt="Rex AI Assistant logo" />
</p>

<p align="center">
  <img src="https://github.com/Blueibear/rex-ai-assistant/actions/workflows/ci.yml/badge.svg" alt="CI status" />
</p>

🧠 Rex AI Assistant

Rex is a local-first, voice-driven AI companion that runs entirely on your machine. It combines:

🔊 Wake-word detection via openWakeWord

🗣️ Speech-to-text using Whisper

🤖 Transformer-based responses via Hugging Face or OpenAI (optional)

🔉 Text-to-speech via Coqui XTTS, with voice cloning support

🌐 Pluggable web search via SerpAPI or DuckDuckGo

🔐 Flask APIs with authentication and rate limiting

🧠 Per-user memory profiles for personalization

✅ Built-in tests and GitHub Actions CI

🏗️ **Modular rex.* namespace** for clean imports and no circular dependencies

Everything runs offline by default — no cloud access unless explicitly enabled.

✨ Highlights

🔊 Customizable wake-word detection (ONNX or fallback hey_jarvis)

🗣️ Speech transcription via Whisper

🤖 Chat model (local Transformers with offline fallback, or OpenAI if configured)

🔉 XTTS TTS with optional voice cloning

🌐 Search plugin with SerpAPI or DuckDuckGo

🔐 HTTP APIs with API key / Cloudflare Access support

🧠 Per-user memory with preferences and history

✅ CI & tests via GitHub Actions

🚀 Quick Start
🔧 Prerequisites

Python 3.10+

Git

FFmpeg (must be in PATH)

Microphone & speakers

(Optional) NVIDIA GPU with CUDA for performance

🧱 Setup
git clone https://github.com/Blueibear/rex-ai-assistant.git
cd rex-ai-assistant

python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate

pip install --upgrade pip
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -r requirements-ml.txt || true  # optional extras (Whisper, TTS)
pip install -r requirements-dev.txt || true  # pytest, coverage, async fixtures
Health Check
python scripts/doctor.py
Runs a quick check for ffmpeg, torch, env vars, and rate limiter. (ffmpeg, torch, env vars).


✅ Or use the helper:

python install.py --with-ml --with-dev

⚡ GPU Acceleration (Optional)
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

🔊 Wake-word Setup

If rex.onnx exists at project root, it is used. Otherwise, the default phrase hey_jarvis is activated.

Train or record:

python record_wakeword.py


On first run, confirmation tones and placeholder voices are synthesized automatically — no binary WAVs are tracked in Git.

🧠 Personalize Voices

Update Memory/<user>/core.json:

"voice": {
  "sample_path": "path/to/your_voice.wav",
  "gender": "male",
  "style": "friendly and warm"
}


If your voice sample is missing, Rex falls back to the XTTS default.

🎙️ Start the Assistant
python rex_assistant.py


Say your wake word, speak your request, and Rex will respond out loud. Press Enter or Ctrl+C to exit.

🎛 Audio Config

List/select devices:

python audio_devices.py --list
python audio_devices.py --set-input <device_id>
python audio_devices.py --set-output <device_id>

🌐 TTS HTTP API (Optional)
python rex_speak_api.py


POST /speak

{
  "text": "Hello Rex",
  "user": "james"
}


Headers:

Content-Type: application/json

X-API-Key: your_secret

Returns: WAV audio.

🔒 Environment Variables
Variable	Purpose
REX_ACTIVE_USER	Default memory profile
REX_WAKEWORD, REX_WAKEWORDS	Wake phrase(s)
REX_WAKEWORD_THRESHOLD	Detection sensitivity
REX_WHISPER_MODEL	Whisper model (tiny, base, ...)
REX_LLM_MODEL, REX_LLM_MAX_TOKENS, REX_LLM_TEMPERATURE	LLM config
OPENAI_API_KEY	Use OpenAI Chat API if REX_LLM_MODEL starts with openai:
REX_INPUT_DEVICE	Input device ID
SERPAPI_KEY	Enables SerpAPI search
REX_SPEAK_API_KEY	Required by /speak endpoint
REX_PROXY_TOKEN	Auth token for Flask proxy
REX_PROXY_ALLOW_LOCAL	Allow local dev bypass (1)
REX_SPEAK_RATE_LIMIT	Requests allowed per window for /speak
REX_SPEAK_RATE_WINDOW	Window size in seconds for rate limiting
REX_SPEAK_MAX_CHARS	Maximum text length accepted by /speak
REX_SPEAK_STORAGE_URI	Limiter storage backend (e.g. redis://localhost:6379/0)
REX_LLM_PROVIDER	Preferred backend (transformers, openai, dummy)
(Rate limiting defaults use in-memory storage. For multi-instance deployments configure a Flask-Limiter backend, or disable limits explicitly.)
dY� Production Deployment Notes

- Serve Flask apps behind a real WSGI stack (gunicorn/uvicorn) and terminate TLS at a reverse proxy such as nginx or Cloudflare.
- Set REX_SPEAK_STORAGE_URI to a shared backend (Redis, Memcached) when running multiple workers so rate limiting stays consistent.
- Store REX_SPEAK_API_KEY in your secret manager and rotate it regularly; restart the speak API after each rotation.
- Monitor /health and application logs to catch TTS or rate limit failures early.
- When packaging Rex as a Python distribution, use the rex.* imports (for example, rex.memory_utils) so modules resolve from the installed package.
🧠 Memory & Personalization

Each user has:

Memory/<user>/
├── core.json     # preferences, voice
├── history.log   # prior chats
└── notes.md      # general info


Profiles support:

Email alias resolution

Freeform notes

Conversation trimming + transcript export

🛠️ Tools

record_wakeword.py — record your own trigger phrase

wakeword_listener.py — test detection

rex_speak_api.py — run standalone TTS server

flask_proxy.py — proxy for Cloudflare Access deployments

manual_whisper_demo.py — test STT with local file

plugins/web_search.py — search backend

🧪 Tests & CI

Run locally:

pytest
# (requires optional dev dependencies)


GitHub Actions runs .github/workflows/ci.yml on every push or PR. Includes:

Linting

Torch compatibility

Memory utils

Flask APIs

Plugin integration

📦 Docker (Optional)

To containerize Rex:

Base image: nvidia/cuda:11.8-runtime

Install: ffmpeg, libsndfile, portaudio19-dev, nvidia-cuda-toolkit

Use torch via cu118 wheel

Expose TTS and proxy ports

📄 License

Released under the MIT License
.

See [CHANGELOG](CHANGELOG.md) for release history.
