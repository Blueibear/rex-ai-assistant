# 🧠 Rex AI Assistant

Rex is a **local-first**, voice-driven AI companion that runs entirely on your machine.

It combines:

- 🔊 **Wake-word detection** via [openWakeWord](https://github.com/dscripka/openWakeWord)  
- 🗣️ **Speech-to-text** using [Whisper](https://github.com/openai/whisper)  
- 🤖 **Transformer-based response generation** (defaults to `distilgpt2`, customizable via Hugging Face)  
- 🔉 **Text-to-speech** via [Coqui XTTS](https://github.com/coqui-ai/TTS), with voice cloning support  
- 🌐 **Pluggable web search** via SerpAPI or DuckDuckGo  
- 🔐 **Flask TTS API endpoint** with authentication  
- 🧠 **Per-user memory profiles** for personalization  

Everything runs **offline by default** — no cloud access unless you explicitly enable it.

## ✨ Highlights

- 🔊 Customizable **wake-word detection** using ONNX models (`rex.onnx` or fallback: `hey_jarvis`)  
- 🗣 Whisper-based **speech transcription**  
- 🤖 Transformer-based **chat model**, configurable via env vars  
- 🔉 XTTS-based **text-to-speech** with optional voice cloning  
- 🌐 Search plugin using SerpAPI (fallback to DuckDuckGo scraping)  
- 🔐 **Local HTTP API** for TTS, with optional token or Cloudflare Access  
- ✅ **CI & tests** run automatically on each commit  
- 🧠 Personalized **user memory**, including notes, conversation history, and more  

## 🚀 Quick Start

### 🔧 Prerequisites

- Python **3.10+**  
- Git  
- FFmpeg (must be in PATH)  
- Microphone & speakers  
- (Optional) NVIDIA GPU with CUDA for speedups

### 🧱 Setup

```bash
git clone https://github.com/Blueibear/rex-ai-assistant.git
cd rex-ai-assistant

python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -r requirements-ml.txt || true
pip install -r requirements-dev.txt || true
```

> ✅ Or run the helper:
> ```bash
> python install.py --with-ml --with-dev
> ```

### ⚡ Enable GPU Acceleration (Optional)

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

Then test:

```python
import torch
print(torch.__version__)              # ends with +cu118
print(torch.cuda.is_available())      # should be True
```

### 🔊 Wake-word Setup

Rex looks for a `rex.onnx` wake-word model at the project root. If not found, it uses the built-in `hey_jarvis` phrase.

Record or train a custom wake word:

```bash
python record_wakeword.py
```

### 🎙️ Run the Assistant

```bash
python rex_assistant.py
```

Say your wake word to begin! Press **Enter** or **Ctrl+C** to exit.

## 🔊 Audio Configuration

List and select audio devices:

```bash
python audio_devices.py --list
python audio_devices.py --set-input <device_id>
python audio_devices.py --set-output <device_id>
```

## 🌐 TTS HTTP API (Optional)

```bash
python rex_speak_api.py
```

Send requests:

```http
POST /speak
Content-Type: application/json
X-API-Key: your_secret

{
  "text": "Hello Rex",
  "user": "james"
}
```

Returns: WAV audio response.

## ⚙️ Configuration via Environment Variables

| Variable | Purpose |
|---------|---------|
| `REX_ACTIVE_USER` | Default profile (e.g. `james`) |
| `REX_WAKEWORD` | Custom wake phrase |
| `REX_WAKEWORD_THRESHOLD` | Wake-word sensitivity (default: `0.5`) |
| `REX_WHISPER_MODEL` | Whisper model size (`tiny`, `base`, ...) |
| `REX_LLM_MODEL` | Transformer model (`distilgpt2`, etc) |
| `REX_LLM_MAX_TOKENS`, `REX_LLM_TEMPERATURE` | LLM generation tuning |
| `SERPAPI_KEY` | Enables web search via SerpAPI |
| `REX_SPEAK_API_KEY` | Token for API protection |
| `REX_PROXY_TOKEN` | Auth token for Flask proxy |
| `REX_PROXY_ALLOW_LOCAL` | Allow localhost bypass (`1` for dev) |

## 🧠 Memory & Profiles

Each user has a folder: `Memory/<username>/`

Contents:

- `core.json` — structured preferences and voice settings  
- `history.log` — conversation history  
- `notes.md` — free-form text notes  
- Optional: voice sample (WAV)

User profiles allow personalized interactions and voice cloning.

## 🛠️ Tools

- `record_wakeword.py` – record/train your own wake-word  
- `wakeword_listener.py` – test wake-word detection independently  
- `rex_speak_api.py` – run standalone TTS HTTP server  
- `flask_proxy.py` – proxy support for Cloudflare Access  
- `plugins/` – drop-in plugin support (e.g. `web_search`)

## 🧪 Tests & CI

Run locally:

```bash
pytest
```

CI runs on every `push` and `pull_request`. Workflow defined in:

```
.github/workflows/ci.yml
```

Includes:

- System + Python deps
- Torch (CPU/GPU)
- All tests with coverage

## 🐳 Docker (Optional)

To containerize Rex:

- Use `nvidia/cuda:11.8-runtime` base
- Install `ffmpeg`, `libsndfile`, `portaudio19-dev`, `nvidia-cuda-toolkit`
- Install PyTorch via `--index-url https://download.pytorch.org/whl/cu118`
- Expose ports for TTS / Proxy as needed

## 📄 License

Released under the [MIT License](LICENSE).

