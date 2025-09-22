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
- 🗃️ **Conversation memory** that compresses older exchanges so prompts stay focused while still honouring long-term context.
- 🌐 **Pluggable web search** that uses SerpAPI when available and gracefully falls back to DuckDuckGo scraping.
- 🔐 **Hardened APIs** including a rate-limited, API-key protected `/speak` endpoint and a Flask proxy ready for Cloudflare Access or shared-secret deployments.
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

> **Note:** If PyTorch or `transformers` cannot be installed on your
> hardware, Rex falls back to an offline acknowledgement mode so you can
> still exercise the wake-word, STT, and TTS pipeline.

### 2. Prepare wake-word & voices

Rex looks for a custom `rex.onnx` wake-word model in the repository root.
The repository intentionally does **not** ship a binary wake-word file; if the
model is missing or empty the assistant automatically falls back to the bundled
openWakeWord keyword `hey_jarvis`. To train or record a personal wake-word
model, run:

```bash
python record_wakeword.py
```

For personalised text-to-speech, add a short WAV sample to your memory
profile (see `Memory/<user>/core.json`). On first run the assistant generates a
synthetic placeholder voice at `assets/voices/placeholder.wav` so the default
profiles work out of the box without storing binary assets in Git. Replace that
file path with your own recording for richer cloning. If the configured sample
is missing, Rex automatically falls back to the default XTTS speaker.

Similarly, the wake-word acknowledgment chirp is generated into
`assets/wake_acknowledgment.wav` the first time you run either
`rex_assistant.py` or `wakeword_listener.py`, so fresh clones never need to
check in binary WAV files just to play the confirmation tone.

Need a quick transcription test? Supply your own clip to
`manual_whisper_demo.py`:

```bash
python manual_whisper_demo.py path/to/recording.wav
```

This keeps the repository free of binary test audio while still letting you
verify Whisper locally.

### 3. Launch the assistant

```bash
python rex_assistant.py
```

Say the configured wake word, speak your request, and Rex will transcribe,
consult memory, generate an answer, and reply out loud. Press **Enter** to exit.
On startup the assistant lists available microphone devices so you can switch
inputs without editing configuration files. Set `REX_INPUT_DEVICE` to skip the
interactive prompt in headless or automated environments.

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
- `REX_WAKEWORDS` – optional comma-separated list of wake phrases to
  monitor simultaneously.
- `REX_WAKEWORD_THRESHOLD` – sensitivity threshold for wake-word detection
  (default `0.5`).
- `REX_WHISPER_MODEL` – Whisper size to load (`tiny`, `base`, `small`, …).
- `REX_LLM_MODEL`, `REX_LLM_MAX_TOKENS`, `REX_LLM_TEMPERATURE` – tune the
  transformer-based response generator. Prefix the model with `openai:` (for
  example `openai:gpt-4o-mini`) to route prompts through the OpenAI Chat
  Completions API instead of a local transformer. Set `OPENAI_API_KEY` and
  install the `openai` package to enable this mode.
- `REX_INPUT_DEVICE` – preset the microphone index when running the assistant
  non-interactively.
- `REX_MEMORY_MAX_BYTES` – refuse to load oversized memory profiles to prevent
  runaway JSON growth (defaults to 128 KiB).

### Web search plugin

- `SERPAPI_KEY` – enables SerpAPI in `plugins/web_search.py`.
- `SERPAPI_ENGINE` – optional SerpAPI engine override (defaults to `google`).

Without a SerpAPI key, the plugin scrapes DuckDuckGo HTML results. Set
Rex's environment variables accordingly if you need to disable web
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

### Speech API security

- `REX_SPEAK_API_KEY` – required shared secret for the `/speak` endpoint.
- `REX_SPEAK_RATE_LIMIT` / `REX_SPEAK_RATE_WINDOW` – limit how many requests
  each caller may issue within a rolling window (defaults to 30 requests per 60
  seconds).
- `REX_SPEAK_MAX_CHARS` – cap the maximum length of accepted text.

## 🧠 Memory & personalisation

Each memory profile lives under `Memory/<user>/` and includes:

- `core.json` – structured profile metadata
- `history.log` – chronological conversation history
- `notes.md` – free-form notes

Update these files to teach Rex about new preferences or to provide voice
samples. The helper functions in `memory_utils.py` normalise profile paths and
email aliases so you can reference a user by folder name, email address, or
profile display name. Runtime conversations are captured by a summarising
`ConversationMemory` helper so prompts stay succinct even after lengthy chats.

## 🛠️ Additional tools

- `wakeword_listener.py` – standalone script that listens for the wake word and plays a confirmation sound when detected.
- `rex_speak_api.py` – HTTP API for text-to-speech only workflows.
- `flask_proxy.py` – reverse proxy suitable for Cloudflare Access deployments.

## 🧪 Test matrix

The repository ships with unit tests covering the language-model wrapper,
memory utilities, and Flask proxy. A lightweight integrity test also guards
against accidentally committing regenerated wake-tone or voice WAV files.
GitHub Actions executes the full matrix on every push so new changes must keep
the suite green.

```bash
pytest
```

## 📄 License

Rex AI Assistant is released under the [MIT License](LICENSE).
