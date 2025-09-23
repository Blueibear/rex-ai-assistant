# 🧠 Rex AI Assistant

Rex is a local-first, voice-driven AI companion that combines wake-word
spotting, OpenAI Whisper transcription, transformer-backed text
responses, and Coqui XTTS speech synthesis. Each user has a dedicated
memory profile so the assistant can tailor answers without sending data
to the cloud.

## ✨ Highlights

- 🔊 **Wake-word detection** via [openWakeWord](https://github.com/dscripka/openWakeWord) with support for custom ONNX models.
- 🗣️ **Speech-to-text** powered by [Whisper](https://github.com/openai/whisper).
- 🧠 **Transformer responses** produced with Hugging Face models (defaults to `distilgpt2`).
- 🔉 **Text-to-speech** using [Coqui XTTS v2](https://github.com/coqui-ai/TTS) with optional per-user voice cloning.
- 🌐 **Pluggable web search** that uses SerpAPI when available and gracefully falls back to DuckDuckGo scraping.
- 🔐 **Local Flask proxy** with shared-secret or Cloudflare Access authentication.
- ✅ **Automated tests & CI** so regressions are caught early.

## 🚀 Quick start

### Prerequisites

- Python **3.10 or newer**
- Git
- FFmpeg (required by Whisper/XTTS for audio conversions)
- Speakers and a microphone

### 1. Clone and install

```bash
git clone https://github.com/Blueibear/rex-ai-assistant.git
cd rex-ai-assistant

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
# Install PyTorch CPU wheels first for reliable installs on Linux
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Prepare wake-word & voices

Rex looks for a custom `rex.onnx` wake-word model in the repository root.
If that file is missing or empty, the assistant automatically falls back
to the bundled openWakeWord keyword `hey_jarvis`. To train or record a
personal wake-word model, run:

```bash
python record_wakeword.py
```

For personalised text-to-speech, add a short WAV sample to your memory
profile (see `Memory/<user>/core.json`). If the file path is invalid,
Rex uses the default XTTS speaker instead.

### 3. Launch the assistant

```bash
python rex_assistant.py
```

Say the configured wake word, speak your request, and Rex will transcribe,
consult memory, generate an answer, and reply out loud. Press **Enter**
to exit.

### 4. Run automated tests

```bash
pytest
```

Continuous integration runs the same test suite on every push via the
GitHub Actions workflow defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## ⚙️ Configuration

Rex reads configuration from environment variables so you can adapt the
assistant to different environments without editing source files.

### Core options

- `REX_ACTIVE_USER` – selects the default memory folder (for example `james`).
- `REX_WAKEWORD` – desired wake phrase. If a matching ONNX model is not
  available, Rex falls back to the keyword specified by
  `REX_WAKEWORD_KEYWORD` (defaults to `hey_jarvis`).
- `REX_WAKEWORD_THRESHOLD` – sensitivity threshold for wake-word detection
  (default `0.5`).
- `WHISPER_MODEL` (or legacy `REX_WHISPER_MODEL`) – Whisper size to load
  (`tiny`, `base`, `small`, `medium`, `large`). The default is `medium` to
  match GPU-equipped deployments.
- `WHISPER_DEVICE` (or legacy `REX_WHISPER_DEVICE`) – device for Whisper
  inference (`cuda`, `cpu`, etc.). Set to `cuda` for GPU acceleration or
  override with `cpu` on systems without CUDA.
- `REX_LLM_MODEL`, `REX_LLM_MAX_TOKENS`, `REX_LLM_TEMPERATURE` – tune the
  transformer-based response generator.

### Web search plugin

- `SERPAPI_KEY` – enables SerpAPI in `plugins/web_search.py`.
- `SERPAPI_ENGINE` – optional SerpAPI engine override (defaults to `google`).

Without a SerpAPI key, the plugin scrapes DuckDuckGo HTML results. Set
`Rex`'s environment variables accordingly if you need to disable web
access entirely.

### Flask proxy authentication

- `REX_PROXY_TOKEN` – shared secret for clients that cannot use Cloudflare Access.
- `REX_PROXY_ALLOW_LOCAL` – set to `1` during development to allow
  unauthenticated requests from `127.0.0.1`.

When requests arrive with the `Cf-Access-Authenticated-User-Email`
header, the proxy resolves that email via `users.json` and loads the
matching memory profile. The `/whoami` route now returns a sanitised
summary instead of the entire memory object to prevent sensitive data
leaks.

## 🧠 Memory & personalisation

Each memory profile lives under `Memory/<user>/` and includes:

- `core.json` – structured profile metadata
- `history.log` – chronological conversation history
- `notes.md` – free-form notes

Update these files to teach Rex about new preferences or to provide
voice samples. The helper functions in `memory_utils.py` normalise
profile paths and email aliases so you can reference a user by folder
name, email address, or profile display name.

## 🛠️ Additional tools

- `wakeword_listener.py` – standalone script that listens for the wake word and plays a confirmation sound when detected.
- `rex_speak_api.py` – HTTP API for text-to-speech only workflows.
- `flask_proxy.py` – reverse proxy suitable for Cloudflare Access deployments.

## 🧪 Test matrix

The repository ships with unit tests covering the language-model wrapper,
memory utilities, and Flask proxy. GitHub Actions executes them on every
push so new changes must keep the suite green.

```bash
pytest
```

## 📄 License

Rex AI Assistant is released under the [MIT License](LICENSE).
