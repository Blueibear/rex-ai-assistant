# Rex AI Assistant — Consolidated Instruction Manual

This manual combines the repository's setup, configuration, operation, and troubleshooting guidance into one practical document. It is intended to be the fastest way to understand how to install, configure, run, and maintain Rex without hopping between multiple docs.

---

## 1. What Rex Is

Rex is a local-first AI assistant with:

- text chat and voice interaction,
- wake word detection,
- Whisper-based speech-to-text,
- multiple LLM backends,
- multiple text-to-speech backends,
- optional integrations such as GitHub, Home Assistant, search, messaging, notifications, and workflow automation.

The project is Python-based, exposes a `rex` CLI, and also includes separate entrypoints for the voice loop, GUI, TTS API, Flask proxy, and diagnostics.

---

## 2. Supported Platforms and Baseline Requirements

### Operating systems

- Windows 10/11
- macOS 11+
- Ubuntu 20.04+

### Python

- Supported: Python 3.9 through 3.13
- Recommended: Python 3.10+
- Best documented Windows path: Python 3.11+

### Required system tools

- `ffmpeg` on `PATH`
- `git`
- microphone and speakers for voice mode

### Optional hardware

- NVIDIA GPU with CUDA 11.8+ or 12.4 for accelerated ML/TTS/STT workflows

---

## 3. How Configuration Is Organized

Rex uses a **dual-config model**:

### `.env` = secrets only
Use `.env` for:

- API keys
- tokens
- passwords
- auth secrets

Examples:

- `OPENAI_API_KEY`
- `REX_SPEAK_API_KEY`
- `HA_TOKEN`
- `REX_PROXY_TOKEN`
- `REX_DASHBOARD_PASSWORD`

### `config/rex_config.json` = runtime settings
Use `config/rex_config.json` for:

- audio devices
- wake word settings
- model/provider selection
- runtime behavior
- profile selection
- non-secret integration configuration

Examples:

- active profile
- input/output device indexes
- wake word keyword and threshold
- LLM provider/model
- STT model/device
- TTS provider/voice

### Important rule

Do **not** put secrets into `config/rex_config.json`. Keep secrets in `.env` only.

---

## 4. Recommended Installation Paths

## 4.1 Fastest dev-friendly install (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
cp .env.example .env
pip install .
pip install -r requirements-cpu.txt
python scripts/doctor.py
```

## 4.2 Fastest dev-friendly install (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
Copy-Item .env.example .env
pip install .
pip install -r requirements-cpu.txt
python scripts/doctor.py
```

## 4.3 GPU install

### Windows / CUDA 12.4

```powershell
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-gpu-cu124.txt
python -c "import torch; print(torch.cuda.is_available())"
```

### Linux / CUDA 11.8

```bash
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-gpu.txt
python -c "import torch; print(torch.cuda.is_available())"
```

## 4.4 Optional extras

```bash
pip install -e .[dev]
pip install -e .[test]
pip install -e .[ml,audio]
pip install -e .[full]
```

### Notes

- Use the `requirements-gpu*.txt` files for GPU installs instead of trying to use GPU extras.
- Base install is lighter and suitable for development.
- CPU requirements are the safest default if you mainly want to verify the app and docs.

---

## 5. Platform-Specific System Dependencies

## Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
  ffmpeg \
  libsndfile1 \
  libasound2-dev \
  portaudio19-dev \
  python3-dev \
  python3-venv
```

## macOS

```bash
brew install ffmpeg portaudio python@3.11
```

## Windows

Install:

- Python 3.11+
- Git
- FFmpeg
- NVIDIA drivers if using GPU

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 6. First-Time Setup Checklist

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install base dependencies.
4. Install either CPU or GPU requirements.
5. Copy `.env.example` to `.env`.
6. Configure your secrets in `.env`.
7. Review `config/rex_config.json` and set runtime defaults.
8. Run `python scripts/doctor.py`.
9. Start with text mode first.
10. Move to voice mode, GUI, and APIs after the health check passes.

---

## 7. Minimum Configuration You Actually Need

For the simplest local setup:

### In `.env`

```env
REX_ACTIVE_USER=default
REX_SPEAK_API_KEY=change-me
```

If using OpenAI:

```env
OPENAI_API_KEY=sk-...
```

### In `config/rex_config.json`

Make sure these areas are sensible:

- `audio`
- `wake_word`
- `models`
- `runtime`
- `active_profile`

### Typical model/runtime choices

- `models.llm_provider`: `transformers`, `openai`, or `ollama`
- `models.stt_model`: `tiny`, `base`, `small`, `medium`, `large`
- `models.stt_device`: `cpu` or `cuda`
- `models.tts_provider`: `xtts`, `edge`, `piper`, `pyttsx3`

---

## 8. Recommended Bring-Up Order

If you are new to the repo, use this order:

### Step 1: Diagnose environment

```bash
python scripts/doctor.py
```

### Step 2: Run text chat

```bash
python -m rex
```

Alternative:

```bash
python rex_assistant.py
```

### Step 3: Configure audio devices if needed

```bash
python audio_config.py --list
python audio_config.py --show
```

### Step 4: Run voice loop

```bash
python rex_loop.py
```

### Step 5: Launch GUI

```bash
python gui.py
```

### Step 6: Run optional services

```bash
python rex_speak_api.py
python flask_proxy.py
```

This sequence narrows failures faster than trying to start everything at once.

---

## 9. Main Ways to Run Rex

## 9.1 Text chat mode

```bash
python -m rex
```

or:

```bash
python rex_assistant.py
```

Use this first because it avoids microphone, wake word, and playback variables.

## 9.2 Voice assistant mode

```bash
python rex_loop.py
```

Useful options:

```bash
python rex_loop.py --user james
python rex_loop.py --enable-plugin web_search
```

### What to expect

1. Rex initializes models.
2. You say the wake word.
3. Rex records your command.
4. STT transcribes it.
5. The LLM generates a response.
6. TTS speaks it back if the TTS stack is available.

## 9.3 GUI mode

```bash
python gui.py
```

The GUI provides:

- Dashboard tab
- Settings tab
- visual editing of environment variables
- backup/restore support for `.env`
- restart indicators for settings changes

## 9.4 TTS API

```bash
python rex_speak_api.py
```

Default documented endpoint pattern:

```text
POST /speak
```

Basic curl example:

```bash
curl -X POST http://localhost:5000/speak \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"text": "Hello from Rex", "user": "default"}' \
  --output speech.wav
```

## 9.5 Flask proxy / dashboard

```bash
python flask_proxy.py
```

The runbook treats this as the main network-facing process, with health endpoints such as:

```bash
curl http://localhost:5000/health/live
curl http://localhost:5000/health/ready
```

---

## 10. Operational Model

The repo documentation describes up to three independently startable processes:

| Process | Entry point | Typical purpose |
|---|---|---|
| Flask proxy | `python flask_proxy.py` | API + dashboard + health endpoints |
| TTS API | `python rex_speak_api.py` | text-to-speech HTTP service |
| Voice loop | `python rex_loop.py` | local wake-word voice assistant |

### Basic start sequence

```bash
python -m rex.migrations apply
python flask_proxy.py
python rex_speak_api.py
python rex_loop.py
```

Start only the pieces you need.

---

## 11. Audio and Wake Word Setup

## List devices

```bash
python audio_config.py --list
```

## Set devices

```bash
python audio_config.py --set-input 1
python audio_config.py --set-output 2
python audio_config.py --show
```

## Wake word configuration

Wake-word runtime settings belong in `config/rex_config.json`, not `.env`.

Important wake word fields include:

- `backend`
- `keyword`
- `wakeword` (legacy alias)
- `threshold`
- `model_path`
- `embedding_path`
- `fallback_to_builtin`
- `fallback_keyword`

### Supported wake-word backend patterns

- built-in OpenWakeWord keyword
- custom ONNX model
- custom embedding model

### Validate a custom wake word file

```bash
python scripts/validate_wakeword_model.py --backend custom_onnx --model-path models/wakewords/hey_rex.onnx
python scripts/validate_wakeword_model.py --backend custom_embedding --embedding-path models/wakewords/hey_rex.pt
```

---

## 12. LLM, STT, and TTS Choices

## LLM providers

- `transformers`
- `openai`
- `ollama`

## STT

Whisper sizes:

- `tiny`
- `base`
- `small`
- `medium`
- `large`

## TTS providers

- `xtts`
- `edge`
- `piper`
- `pyttsx3`

### Practical advice

- Start with CPU + smaller models to validate the pipeline.
- Move to GPU only after the base setup works.
- If voice mode fails but text mode works, check the TTS installation and playback library separately.

### Known TTS/pipeline caveat from repo instructions

A documented breakpoint exists when Coqui XTTS is not installed:

- the pipeline can reach `llm_response_received`,
- then fail before `tts_input_prepared`,
- because `_get_tts()` raises `TextToSpeechError("TTS is not installed")`,
- and the error is currently logged and swallowed by the conversation loop.

Also documented: on non-Windows systems without `simpleaudio`, audio may be saved without being played even though later pipeline stages are logged.

---

## 13. Useful CLI Commands and Features

## Diagnostics

```bash
rex doctor
python -m rex doctor
python scripts/doctor.py
```

## Tool registry

```bash
rex tools
rex tools -v
```

## GitHub integration

```bash
rex gh repos
rex gh prs owner/repo
rex gh issue-create owner/repo --title "Bug" --body "Details"
rex gh pr-create owner/repo --head feature-branch --base main --title "Title" --body "Body"
```

## Workflow planning / autonomy

```bash
rex plan "send monthly newsletter"
rex plan "check weather in Dallas" --execute
rex approvals
rex approvals --approve <approval_id>
rex executor resume <workflow_id>
```

## Legacy env migration helper

```bash
rex-config migrate-legacy-env
```

---

## 14. Security and Configuration Rules to Follow

1. Keep secrets in `.env` only.
2. Do not commit `.env`.
3. Prefer localhost binding unless remote access is truly needed.
4. Network-facing endpoints should be authenticated and rate-limited.
5. Use credential-manager flows where the repo already supports them.
6. Avoid logging secrets, tokens, or passwords.
7. Treat external URLs and external inputs as untrusted.
8. Use runtime-guarded imports for optional heavy dependencies.

### Packaging rule

Do not reintroduce GPU extras like `.[gpu-cu124]`. GPU installs are intentionally requirements-file based.

---

## 15. Day-2 Operations and Health Checks

## Health endpoints

```bash
curl -s http://localhost:5000/health/live
curl -s http://localhost:5000/health/ready
```

## Restart pattern

```bash
kill -TERM $(pgrep -f flask_proxy.py)
sleep 2
python flask_proxy.py &
```

## Logs

Typical pattern:

```bash
python flask_proxy.py >> logs/flask_proxy.log 2>&1
```

Useful filters:

```bash
grep -E "ERROR|CRITICAL" logs/flask_proxy.log
grep "req-abc123" logs/flask_proxy.log
tail -f logs/flask_proxy.log | grep --line-buffered ERROR
```

---

## 16. Troubleshooting by Symptom

## `ffmpeg` missing

Install ffmpeg and ensure it is on `PATH`.

## No API keys configured

Set the relevant secrets in `.env`.

## PyTorch missing or wrong variant

Reinstall with the appropriate requirements file:

```bash
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-cpu.txt
```

or:

```bash
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-gpu-cu124.txt
```

or:

```bash
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-gpu.txt
```

## Wake word not detected

```bash
python audio_config.py --list
python wakeword_listener.py
python record_wakeword.py
```

Consider lowering the threshold in runtime config.

## Windows audio playback limitations

The repo docs note that `simpleaudio` is automatically disabled on Windows due to build issues, so audio playback can be limited even when the rest of the assistant works.

## `speexdsp_ns` missing

The docs state this dependency is disabled for Windows compatibility and noise suppression is turned off by default.

## Rate limiting on TTS API

Increase these secrets/settings in `.env` if appropriate:

```env
REX_SPEAK_RATE_LIMIT=60
REX_SPEAK_RATE_WINDOW=60
REX_SPEAK_STORAGE_URI=redis://localhost:6379/0
```

## CUDA out of memory

Use smaller models or switch to CPU.

---

## 17. What Docs Matter Most

If you want the shortest reliable reading list, use this order:

1. `README.md` — project overview and quick start
2. `INSTALL.md` — setup paths and system dependencies
3. `CONFIGURATION.md` — runtime config structure
4. `docs/configuration.md` — environment variable reference
5. `docs/usage.md` — key run commands
6. `docs/runbook.md` — operations and health checks
7. `docs/troubleshooting.md` — common failure recovery
8. `README.windows.md` — Windows-specific quickstart

### Docs that appear historical or specialized

Some docs in `docs/` are targeted to stabilization, audits, or specific feature areas. They are useful, but they should not be treated as the first source for everyday setup unless your task specifically touches those subsystems.

---

## 18. Best-Practice Setup for Most Users

If you just want the safest path to a working Rex instance:

1. Install base dependencies in a clean virtual environment.
2. Install `requirements-cpu.txt` first.
3. Copy `.env.example` to `.env` and set only the secrets you actually need.
4. Keep runtime settings in `config/rex_config.json`.
5. Run `python scripts/doctor.py`.
6. Verify `python -m rex` works.
7. Configure audio devices.
8. Verify `python rex_loop.py`.
9. Add optional APIs and integrations only after the local path works.
10. Use GPU requirements only after the CPU path is stable.

---

## 19. One-Page Quickstart

### macOS/Linux

```bash
git clone https://github.com/Blueibear/askrex-assistant.git
cd rex-ai-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
cp .env.example .env
pip install .
pip install -r requirements-cpu.txt
python scripts/doctor.py
python -m rex
```

### Windows PowerShell

```powershell
git clone https://github.com/Blueibear/askrex-assistant.git
cd rex-ai-assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
Copy-Item .env.example .env
pip install .
pip install -r requirements-cpu.txt
python scripts/doctor.py
python -m rex
```

### Then progress to

```bash
python audio_config.py --list
python rex_loop.py
python gui.py
python rex_speak_api.py
python flask_proxy.py
```

---

## 20. Final Notes

- Start simple: text mode before voice mode.
- Treat `.env` as secrets-only.
- Treat `config/rex_config.json` as the source of runtime truth.
- Prefer CPU install first, GPU second.
- Use `doctor` before debugging deeper.
- Expect some docs in the repo to be subsystem-specific or historical; this manual is meant to be the practical starting point.
