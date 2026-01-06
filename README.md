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

| Component | Requirement |
|-----------|-------------|
| **OS** | macOS 11+, Windows 10+, or Ubuntu 20.04+ |
| **Python** | 3.9 or newer (3.10+ recommended) |
| **FFmpeg** | Must be installed and available on PATH |
| **Hardware** | Microphone and speakers for voice mode |
| **GPU** (optional) | NVIDIA GPU with CUDA 11.8+ for acceleration |

## Quickstart

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
```

### Windows (PowerShell)

```powershell
# Clone repository
git clone https://github.com/Blueibear/rex-ai-assistant.git
cd rex-ai-assistant

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Copy environment template
Copy-Item .env.example .env
# Edit .env with your preferred editor to set API keys and options

# Install dependencies (CPU-only PyTorch)
pip install --upgrade pip setuptools wheel
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 `
  --index-url https://download.pytorch.org/whl/cpu
pip install -e .

# Run health check
python scripts/doctor.py

# Start text-based chat mode
python -m rex

# Or start full voice assistant with wake word
python rex_loop.py
```

### Using the Interactive Installer

Rex includes an interactive installer that automates setup:

```bash
# Basic installation
python install.py

# Include ML models (Whisper, XTTS)
python install.py --with-ml

# Include development tools (pytest, ruff, black, mypy)
python install.py --with-dev

# Auto-install ffmpeg (Linux/macOS only)
python install.py --auto-install-ffmpeg

# Test audio devices
python install.py --mic-test
```

### GPU Acceleration (Optional)

For NVIDIA GPUs with CUDA 11.8:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1+:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Configuration (Environment Variables)

All environment variables can be set in your `.env` file. Copy `.env.example` to `.env` and customize as needed.

### Core Settings

| Variable | Required? | Default | Description | Example |
|----------|-----------|---------|-------------|---------|
| `REX_ACTIVE_USER` | No | `default` | Active user profile (maps to `Memory/<user>/core.json`) | `james` |
| `REX_LOG_LEVEL` | No | `INFO` | Logging verbosity | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `REX_FILE_LOGGING_ENABLED` | No | `true` | Enable file logging (false = stdout only) | `true`, `false` |
| `REX_DEVICE` | No | `cpu` | Device for model inference | `cpu`, `cuda` |

### Wake Word Detection

| Variable | Required? | Default | Description | Example |
|----------|-----------|---------|-------------|---------|
| `REX_WAKEWORD` | No | `rex` | Wake word phrase | `rex`, `jarvis`, `computer` |
| `REX_WAKEWORD_KEYWORD` | No | `hey_jarvis` | openWakeWord model keyword | `hey_jarvis` |
| `REX_WAKEWORD_THRESHOLD` | No | `0.5` | Detection sensitivity (0.0-1.0, higher = stricter) | `0.6` |

### Audio Configuration

| Variable | Required? | Default | Description | Example |
|----------|-----------|---------|-------------|---------|
| `REX_INPUT_DEVICE` | No | (system default) | Microphone device index | `0`, `1`, `2` |
| `REX_OUTPUT_DEVICE` | No | (system default) | Speaker device index | `0`, `1`, `2` |
| `REX_SAMPLE_RATE` | No | `16000` | Audio sample rate (Hz) | `16000` |
| `REX_COMMAND_DURATION` | No | `5.0` | Recording duration (seconds) | `5.0` |

**Tip:** List available audio devices:
```bash
python audio_config.py --list
python audio_config.py --set-input 1
python audio_config.py --set-output 2
```

### Speech Recognition (Whisper)

| Variable | Required? | Default | Description | Example |
|----------|-----------|---------|-------------|---------|
| `REX_WHISPER_MODEL` | No | `base` | Whisper model size | `tiny`, `base`, `small`, `medium`, `large` |
| `REX_WHISPER_DEVICE` | No | `cpu` | Device for Whisper | `cpu`, `cuda` |

**Note:** Larger models are more accurate but slower. `tiny` is fastest, `large` is most accurate.

### Language Model (LLM)

| Variable | Required? | Default | Description | Example |
|----------|-----------|---------|-------------|---------|
| `REX_LLM_PROVIDER` | No | `transformers` | LLM backend | `transformers`, `openai`, `ollama` |
| `REX_LLM_MODEL` | No | `distilgpt2` | Model name or path | `distilgpt2`, `gpt2`, `gpt-3.5-turbo` |
| `REX_LLM_TEMPERATURE` | No | `0.7` | Generation randomness (0.0-2.0) | `0.7` |
| `REX_LLM_MAX_TOKENS` | No | `120` | Maximum response length | `120` |

### OpenAI API (Optional)

| Variable | Required? | Default | Description | Example |
|----------|-----------|---------|-------------|---------|
| `OPENAI_API_KEY` | **Yes** (if using OpenAI) | — | OpenAI API key | `sk-...` |
| `OPENAI_MODEL` | No | `gpt-3.5-turbo` | OpenAI model name | `gpt-4`, `gpt-3.5-turbo` |
| `OPENAI_BASE_URL` | No | (OpenAI default) | Custom API endpoint | `https://api.openai.com/v1` |

### Text-to-Speech (TTS)

| Variable | Required? | Default | Description | Example |
|----------|-----------|---------|-------------|---------|
| `REX_TTS_PROVIDER` | No | `xtts` | TTS backend | `xtts`, `edge`, `pyttsx3` |
| `REX_TTS_MODEL` | No | `tts_models/multilingual/multi-dataset/xtts_v2` | Coqui TTS model name | (see Coqui docs) |
| `REX_TTS_VOICE` | No | `en-US-AndrewNeural` | Edge TTS voice name | `en-US-JennyNeural` |
| `REX_PIPER_MODEL` | No | `voices/en_US-lessac-medium.onnx` | Piper model path | (if using Piper) |
| `REX_SPEAK_LANGUAGE` | No | `en` | Language code | `en`, `es`, `fr`, `de` |

**Tip:** List available edge-tts voices:
```bash
python list_voices.py
```

### Web Search Plugins

| Variable | Required? | Default | Description | Example |
|----------|-----------|---------|-------------|---------|
| `SERPAPI_KEY` | No | — | SerpAPI key | (from serpapi.com) |
| `BRAVE_API_KEY` | No | — | Brave Search API key | (from brave.com/search/api) |
| `GOOGLE_API_KEY` | No | — | Google Custom Search API key | (from console.cloud.google.com) |
| `GOOGLE_CSE_ID` | No | — | Google Custom Search Engine ID | (from cse.google.com) |

### Flask TTS API Security

| Variable | Required? | Default | Description | Example |
|----------|-----------|---------|-------------|---------|
| `REX_SPEAK_API_KEY` | **Yes** (for API) | — | API key for `/speak` endpoint | `your-secret-key-here` |
| `REX_SPEAK_RATE_LIMIT` | No | `30` | Requests allowed per window | `30` |
| `REX_SPEAK_RATE_WINDOW` | No | `60` | Rate limit window (seconds) | `60` |
| `REX_SPEAK_MAX_CHARS` | No | `800` | Maximum text length | `800` |
| `REX_SPEAK_STORAGE_URI` | No | `memory://` | Limiter backend storage | `redis://localhost:6379/0` |
| `REX_ALLOWED_ORIGINS` | No | `http://localhost:*` | CORS origins (comma-separated) | `http://localhost:3000,https://app.example.com` |

### Flask Proxy (Optional)

| Variable | Required? | Default | Description | Example |
|----------|-----------|---------|-------------|---------|
| `REX_PROXY_TOKEN` | No | — | Shared secret for proxy auth | `shared-secret` |
| `REX_PROXY_ALLOW_LOCAL` | No | `0` | Allow local dev without token | `1` (enable), `0` (disable) |

## Usage

### 1. Text-Based Chat Mode

Start an interactive text-based conversation with Rex (no microphone required):

```bash
python -m rex
# or
python rex_assistant.py
```

Type your messages and press Enter. Type `exit` or `quit` to stop.

**Example session:**
```
🎤 Rex assistant ready. Type 'exit' or 'quit' to stop.
You: What is the capital of France?
Rex: The capital of France is Paris.
You: exit
```

### 2. Voice Assistant Mode

Start the full voice assistant with wake word detection:

```bash
python rex_loop.py

# Override user profile
python rex_loop.py --user james

# Enable specific plugins only
python rex_loop.py --enable-plugin web_search
```

**How to use:**
1. Wait for Rex to initialize (models may take 10-30 seconds to load on first run)
2. Say your wake word (default: "rex" or "hey jarvis")
3. Wait for the acknowledgment sound
4. Speak your command within 5 seconds
5. Rex will transcribe, process, and respond with speech
6. Press `Ctrl+C` to exit

### 3. GUI Settings Editor

Rex includes a user-friendly graphical settings editor that lets you configure all environment variables without manually editing `.env` files.

**Launch the GUI:**

```bash
python gui.py
```

**Features:**

- **Dashboard Tab**: Monitor Rex assistant status and recent conversation history
- **Settings Tab**: Visual editor for ALL environment variables
  - Organized by section (Core Settings, Wakeword, Audio, LLM, TTS, etc.)
  - Smart controls: dropdowns for enums, checkboxes for booleans, spinboxes for numbers, path pickers for files
  - Help tooltips: Hover or click the "?" icon next to any setting for detailed explanations
  - Secret masking: API keys and tokens are hidden by default with Show/Hide toggles
  - Restart indicators: Settings that require restart are marked with ⚠ icon
  - Advanced section: Edit custom environment variables not in `.env.example`
  - Add custom keys: Create new environment variables on the fly

**Backup & Restore:**

The Settings editor automatically creates timestamped backups before saving. Use the Backup and Restore buttons to manage your configurations:

- **Save**: Updates `.env` with new values (creates backup first)
- **Reset to Defaults**: Restores all settings to `.env.example` defaults
- **Backup**: Manually create a backup of current `.env`
- **Restore**: Choose from previous backups to restore
- **Open .env in Notepad** (Windows): Power-user option to edit `.env` directly

All backups are stored in the `backups/` directory with timestamps like `.env.backup.20240615_143022`.

**Model Selection:**

The LLM Model field adapts based on your selected provider:
- **Transformers**: Text entry with Browse button for local model paths
- **OpenAI**: Dropdown with common models (gpt-3.5-turbo, gpt-4, etc.)
- **Ollama**: Refresh button to query local Ollama instance for installed models

**Ollama Integration:**

If you're using Ollama as your LLM provider, click the "Refresh" button next to the model dropdown to automatically populate it with models from your local Ollama installation.

### 4. Audio Device Configuration

List and configure audio devices:

```bash
# List all available audio devices
python audio_config.py --list

# Set microphone (input device)
python audio_config.py --set-input 1

# Set speakers (output device)
python audio_config.py --set-output 2

# Show current configured devices
python audio_config.py --show
```

### 5. TTS API Server

Run the standalone text-to-speech HTTP API:

```bash
python rex_speak_api.py
```

**API endpoint:** `POST http://localhost:5000/speak`

**Request:**
```json
{
  "text": "Hello, world!",
  "user": "james"
}
```

**Headers:**
```
Content-Type: application/json
X-API-Key: your-secret-key
```

**Response:** WAV audio file

**Example using curl:**
```bash
curl -X POST http://localhost:5000/speak \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"text": "Hello from Rex!", "user": "default"}' \
  --output speech.wav
```

### 5. Health Check & Diagnostics

Run the doctor script to verify your setup:

```bash
python scripts/doctor.py
```

**Output example:**
```
Rex Doctor
Platform: Darwin 22.6.0 (arm64)

[PASS] Python version: 3.11 detected
[PASS] ffmpeg: ffmpeg available at /opt/homebrew/bin/ffmpeg
[PASS] CUDA: CUDA not detected – running in CPU mode
[WARN] REX_SPEAK_API_KEY: not set (Required for /speak TTS endpoint)
[PASS] OPENAI_API_KEY: set
[PASS] Rate limiter storage: using in-memory backend

Summary: 1 warning(s) detected.
```

### 6. Testing Individual Components

**Test Whisper transcription:**
```bash
python manual_whisper_demo.py path/to/audio.wav --model base
```

**Test web search:**
```bash
python manual_search_demo.py "Python programming tutorials"
```

**Record custom wake word:**
```bash
python record_wakeword.py
```

## Docker

### Build and Run

```bash
# Build image
docker build -t rex-ai-assistant .

# Run with environment file
docker run --rm --env-file .env -it rex-ai-assistant

# Run with volume mounts for persistent data
docker run --rm --env-file .env \
  -v $(pwd)/Memory:/app/Memory \
  -v $(pwd)/transcripts:/app/transcripts \
  -v $(pwd)/models:/app/models \
  -it rex-ai-assistant

# Run TTS API server (expose port 5000)
docker run --rm --env-file .env -p 5000:5000 \
  -it rex-ai-assistant python rex_speak_api.py
```

**Note:** Docker image uses CPU-only PyTorch by default. For GPU support, modify the `Dockerfile` to install CUDA-enabled PyTorch wheels.

## Memory & Personalization

Each user has a dedicated profile in `Memory/<username>/`:

```
Memory/james/
├── core.json         # User preferences and voice settings
├── history.log       # Conversation history
└── notes.md          # Freeform notes about the user
```

**Example `core.json`:**
```json
{
  "name": "James",
  "email": "james@example.com",
  "preferences": {
    "preferred_name": "Jim",
    "timezone": "America/New_York"
  },
  "voice": {
    "sample_path": "Memory/james/voice_sample.wav",
    "gender": "male",
    "style": "friendly and warm"
  }
}
```

Rex uses voice cloning with XTTS when a valid `voice.sample_path` is provided.

## Development

### Code Quality Tools

```bash
# Activate development dependencies
pip install -e .[dev]

# Lint with Ruff
ruff check .

# Format with Black
black .

# Type check with Mypy
mypy .

# Run all linting/formatting
ruff check . && black --check . && mypy .
```

### Running Tests

```bash
# Install test dependencies
pip install -e .[test]

# Run all tests
pytest

# Run with coverage
pytest --cov=rex --cov-report=html

# Run only unit tests (skip slow/audio/GPU tests)
pytest -m "not slow and not audio and not gpu"

# Run specific test file
pytest tests/test_config.py

# Verbose output
pytest -v
```

### Available Test Markers

- `unit` — Fast unit tests
- `integration` — Tests requiring external services
- `slow` — Tests that take significant time
- `audio` — Tests requiring audio hardware
- `gpu` — Tests requiring GPU acceleration
- `network` — Tests requiring network access

## Troubleshooting

### Missing API Keys

**Error:** `REX_SPEAK_API_KEY: not set`

**Solution:** Set the API key in your `.env` file:
```env
REX_SPEAK_API_KEY=your-secret-key-here
```

### FFmpeg Not Found

**Error:** `ffmpeg executable not found`

**Solution:**
- **macOS:** `brew install ffmpeg`
- **Ubuntu/Debian:** `sudo apt install ffmpeg`
- **Windows:** Download from https://ffmpeg.org/download.html and add to PATH

### PyTorch Installation Issues

**Error:** `torch is not installed`

**Solution:** Install PyTorch manually:
```bash
# CPU-only
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

### Microphone Permissions (macOS)

**Error:** `Audio device not accessible`

**Solution:**
1. Open **System Settings** → **Privacy & Security** → **Microphone**
2. Enable microphone access for **Terminal** or your Python interpreter

### WASAPI Issues (Windows)

**Error:** `sounddevice` or `portaudio` errors on Windows

**Solution:**
1. Install Visual C++ Redistributables: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Install `pyaudio` from wheels: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

### Wake Word Not Detected

**Issue:** Rex doesn't respond to wake word

**Solution:**
1. Check microphone is working: `python audio_config.py --list`
2. Lower threshold: `REX_WAKEWORD_THRESHOLD=0.3` in `.env`
3. Test wake word detection: `python wakeword_listener.py`
4. Record custom wake word: `python record_wakeword.py`

### Rate Limit Errors (TTS API)

**Error:** `429 Too Many Requests`

**Solution:** Increase rate limits in `.env`:
```env
REX_SPEAK_RATE_LIMIT=60
REX_SPEAK_RATE_WINDOW=60
```

For production deployments with multiple workers, use Redis:
```env
REX_SPEAK_STORAGE_URI=redis://localhost:6379/0
```

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
1. Use smaller Whisper model: `REX_WHISPER_MODEL=tiny` or `base`
2. Reduce max tokens: `REX_LLM_MAX_TOKENS=50`
3. Switch to CPU: `REX_DEVICE=cpu` and `REX_WHISPER_DEVICE=cpu`

## Release & Versioning

This project uses **Release Please** for automated versioning and changelog generation.

- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
- Merging to `main` triggers Release Please to create a release PR
- Merging the release PR creates a GitHub release with tags and changelog

**Example commit messages:**
```bash
feat: add support for custom wake words
fix: resolve audio device selection on Windows
docs: update installation guide for GPU setup
chore: bump dependencies to latest versions
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run linting and tests: `ruff check . && black . && pytest`
5. Commit with conventional commits: `git commit -m "feat: add amazing feature"`
6. Push to your fork: `git push origin feature/amazing-feature`
7. Open a Pull Request

## License

Released under the **MIT License**.

Copyright © 2025 James Ramsey

See [LICENSE](LICENSE) for full details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Coqui TTS](https://github.com/coqui-ai/TTS) for text-to-speech
- [openWakeWord](https://github.com/dscripka/openWakeWord) for wake word detection
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for LLM support
- [Flask](https://flask.palletsprojects.com/) for API framework
- [Release Please](https://github.com/googleapis/release-please) for automated releases

---

**Need help?** Check the [Troubleshooting](#troubleshooting) section or file an issue at https://github.com/Blueibear/rex-ai-assistant/issues
