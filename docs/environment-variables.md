# Rex AI Assistant — Environment Variables

Rex uses a dual-configuration system for better security:
- **config/rex_config.json** — Runtime settings (audio, models, wake word, etc.)
- **.env** — Secrets only (API keys, tokens)

Legacy non-secret environment variables (e.g. `OPENAI_BASE_URL`) are ignored at runtime. If any are set, Rex logs a warning. To migrate them into `config/rex_config.json`, run:

```bash
rex-config migrate-legacy-env
```

See [CONFIGURATION.md](../CONFIGURATION.md) for full details including configuration precedence and the no-overwrite migration rule.

All environment variables can be set in your `.env` file. Copy `.env.example` to `.env` and customize as needed.

## Core Settings

| Variable | Required? | Default | Description | Example |
|----------|-----------|---------|-------------|---------|
| `REX_ACTIVE_USER` | No | `default` | Active user profile (maps to `Memory/<user>/core.json`) | `james` |
| `REX_LOG_LEVEL` | No | `INFO` | Logging verbosity | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `REX_FILE_LOGGING_ENABLED` | No | `true` | Enable file logging (false = stdout only) | `true`, `false` |
| `REX_DEVICE` | No | `cpu` | Device for model inference | `cpu`, `cuda` |

## Wake Word Detection

| Variable | Required? | Default | Description | Example |
|----------|-----------|---------|-------------|---------|
| `REX_WAKEWORD` | No | `rex` | Wake word phrase | `rex`, `jarvis`, `computer` |
| `REX_WAKEWORD_KEYWORD` | No | `hey_jarvis` | openWakeWord model keyword | `hey_jarvis` |
| `REX_WAKEWORD_THRESHOLD` | No | `0.5` | Detection sensitivity (0.0-1.0, higher = stricter) | `0.6` |

## Audio Configuration

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

## Speech Recognition (Whisper)

| Variable | Required? | Default | Description | Example |
|----------|-----------|---------|-------------|---------|
| `REX_WHISPER_MODEL` | No | `base` | Whisper model size | `tiny`, `base`, `small`, `medium`, `large` |
| `REX_WHISPER_DEVICE` | No | `cpu` | Device for Whisper | `cpu`, `cuda` |

**Note:** Larger models are more accurate but slower. `tiny` is fastest, `large` is most accurate.

## Language Model (LLM)

| Variable | Required? | Default | Description | Example |
|----------|-----------|---------|-------------|---------|
| `REX_LLM_PROVIDER` | No | `transformers` | LLM backend | `transformers`, `openai`, `ollama` |
| `REX_LLM_MODEL` | No | `distilgpt2` | Model name or path | `distilgpt2`, `gpt2`, `gpt-3.5-turbo` |
| `REX_LLM_TEMPERATURE` | No | `0.7` | Generation randomness (0.0-2.0) | `0.7` |
| `REX_LLM_MAX_TOKENS` | No | `120` | Maximum response length | `120` |

## OpenAI API (Optional)

| Variable | Required? | Default | Description | Example |
|----------|-----------|---------|-------------|---------|
| `OPENAI_API_KEY` | **Yes** (if using OpenAI) | — | OpenAI API key (secret — keep in `.env`) | `sk-...` |

> **Note:** `OPENAI_MODEL` and `OPENAI_BASE_URL` are **not** active runtime environment
> variables. They are ignored at runtime and must be set in `config/rex_config.json` under
> the `openai` section (`openai.model` and `openai.base_url`).
> If you have these in your `.env` from a previous installation, run:
> ```bash
> rex-config migrate-legacy-env
> ```
> to move them into `config/rex_config.json` automatically. See
> [CONFIGURATION.md](../CONFIGURATION.md) for the full precedence and migration guide.

## Text-to-Speech (TTS)

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

## Web Search Plugins

| Variable | Required? | Default | Description | Example |
|----------|-----------|---------|-------------|---------|
| `SERPAPI_KEY` | No | — | SerpAPI key | (from serpapi.com) |
| `BRAVE_API_KEY` | No | — | Brave Search API key | (from brave.com/search/api) |
| `GOOGLE_API_KEY` | No | — | Google Custom Search API key | (from console.cloud.google.com) |
| `GOOGLE_CSE_ID` | No | — | Google Custom Search Engine ID | (from cse.google.com) |

## Flask TTS API Security

| Variable | Required? | Default | Description | Example |
|----------|-----------|---------|-------------|---------|
| `REX_SPEAK_API_KEY` | **Yes** (for API) | — | API key for `/speak` endpoint | `your-secret-key-here` |
| `REX_SPEAK_RATE_LIMIT` | No | `30` | Requests allowed per window | `30` |
| `REX_SPEAK_RATE_WINDOW` | No | `60` | Rate limit window (seconds) | `60` |
| `REX_SPEAK_MAX_CHARS` | No | `800` | Maximum text length | `800` |
| `REX_SPEAK_STORAGE_URI` | No | `memory://` | Limiter backend storage | `redis://localhost:6379/0` |
| `REX_ALLOWED_ORIGINS` | No | `http://localhost:*` | CORS origins (comma-separated) | `http://localhost:3000,https://app.example.com` |

## Flask Proxy (Optional)

| Variable | Required? | Default | Description | Example |
|----------|-----------|---------|-------------|---------|
| `REX_PROXY_TOKEN` | No | — | Shared secret for proxy auth | `shared-secret` |
| `REX_PROXY_ALLOW_LOCAL` | No | `0` | Allow local dev without token | `1` (enable), `0` (disable) |
