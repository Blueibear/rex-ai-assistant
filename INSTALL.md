# AskRex Assistant Installation Guide

This guide covers supported install paths for the current repo. AskRex is Python 3.11-only today; use a 3.11 virtual environment even if your system default `python` is newer.

## System Requirements

| Component | Requirement |
|---|---|
| OS | Windows 10/11, macOS, or Linux |
| Python | 3.11 (`>=3.11,<3.12`) |
| Disk | Several GB if installing ML/TTS models |
| Audio | Microphone and speakers for voice mode |
| FFmpeg | Required for parts of the audio/TTS stack |
| Node.js/npm | Required only for the Electron GUI under `gui/` |
| GPU | Optional NVIDIA CUDA path via the GPU requirements files |

Python 3.12, 3.13, and 3.14 are not supported by the validated ML/TTS dependency path and are rejected by the app.

## Quick Start

Windows PowerShell:

```powershell
git clone https://github.com/Blueibear/AskRex-Assistant.git
cd AskRex-Assistant

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install .

Copy-Item config\rex_config.example.json config\rex_config.json -ErrorAction SilentlyContinue
Copy-Item .env.example .env -ErrorAction SilentlyContinue

python -m rex doctor
python -m rex
```

macOS / Linux:

```bash
git clone https://github.com/Blueibear/AskRex-Assistant.git
cd AskRex-Assistant

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install .

cp -n config/rex_config.example.json config/rex_config.json
cp -n .env.example .env

python -m rex doctor
python -m rex
```

## Install Options

Base install:

```bash
pip install .
```

This installs the Flask/config/security/runtime dependencies plus lightweight TTS backends, but not the heavy ML/audio stack.

CPU ML/audio stack:

```bash
pip install -r requirements-cpu.txt
```

Validated Windows CUDA 12.4 stack:

```bash
pip install -r requirements-gpu-cu124.txt
```

Alternative CUDA 11.8 stack:

```bash
pip install -r requirements-gpu.txt
```

Editable development install:

```bash
pip install -e ".[dev,test]"
```

Full extra from package metadata:

```bash
pip install -e ".[full]"
```

Use the split requirements files for GPU installs because CUDA PyTorch wheels require the PyTorch extra index URL, which cannot be expressed reliably as a normal optional extra.

## Configuration Files

Create these files from the checked-in examples if they do not already exist:

```bash
cp config/rex_config.example.json config/rex_config.json
cp .env.example .env
```

On Windows:

```powershell
Copy-Item config\rex_config.example.json config\rex_config.json -ErrorAction SilentlyContinue
Copy-Item .env.example .env -ErrorAction SilentlyContinue
```

Use:

- `config/rex_config.json` for non-secret runtime settings: audio, wake word, models, profiles, integrations, UI flags, tool settings.
- `.env` for secrets: API keys, tokens, and shared service keys.
- `profiles/*.json` for profile capabilities and runtime overrides.

Useful commands:

```bash
rex-config show
rex-config migrate-legacy-env --dry-run
rex-config migrate-legacy-env
```

## Startup Modes

| Mode | Command | Notes |
|---|---|---|
| Text chat | `rex` or `python -m rex` | Default interactive CLI |
| Diagnostics | `rex doctor` | Environment and dependency checks |
| Voice loop | `python rex_loop.py` | Wake word -> STT -> LLM -> TTS |
| Python web dashboard | `rex-gui` | Opens `http://127.0.0.1:8765/ui/` |
| Electron desktop GUI | `cd gui && npm.cmd run dev` | Requires Node/npm and Python bridges |
| TTS API | `rex-speak-api` | Requires `REX_SPEAK_API_KEY`; default port 5005 |
| OpenClaw tool server | `rex-tool-server` | Requires `REX_TOOL_API_KEY`; default port 18790 |
| Windows computer agent | `rex-agent` | Optional remote PC control agent |

`python run_gui.py` and `python gui.py` are deprecated Tkinter paths and should not be used for normal operation.

Override the Python web dashboard port:

```bash
REX_GUI_PORT=9000 rex-gui
```

PowerShell:

```powershell
$env:REX_GUI_PORT=9000; rex-gui
```

## Electron GUI Setup

```powershell
cd gui
npm.cmd install
npm.cmd run dev
```

Build and preview:

```powershell
cd gui
npm.cmd run typecheck
npm.cmd run build
npm.cmd run preview
```

For Electron-only verification harnesses, build first so `gui/dist-electron/main/index.js` matches the TypeScript sources.

## Platform Notes

Linux:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1 libasound2-dev portaudio19-dev python3-dev python3-venv
```

macOS:

```bash
brew install ffmpeg portaudio python@3.11
```

Windows:

- Install Python 3.11 and use `py -3.11`.
- Install FFmpeg and ensure `ffmpeg.exe` is on `PATH`.
- Native Windows is supported; WSL2 is also usable for CLI/server flows.
- `simpleaudio` is optional and disabled on Windows because it has known build issues.

## Running Tests

```bash
pip install -e ".[dev,test]"
pytest
pytest -m "not slow and not audio and not gpu"
pytest --cov=rex --cov-report=html
```

Electron checks:

```powershell
cd gui
npm.cmd run typecheck
npm.cmd run build
```

## Troubleshooting

If `python -m rex ...` rejects your interpreter, confirm the active Python:

```bash
python --version
```

On Windows, prefer:

```powershell
py -3.11 -m rex doctor
```

Run diagnostics:

```bash
python -m rex doctor
```

Common fixes:

- Missing FFmpeg: install FFmpeg and reopen the terminal so `PATH` updates.
- Missing ML dependencies: install `requirements-cpu.txt` or the matching GPU requirements file.
- OpenAI/Ollama/model failures: check `config/rex_config.json` model settings and `.env` secrets.
- TTS API startup failure: set `REX_SPEAK_API_KEY`.
- Tool server 401s: set `REX_TOOL_API_KEY`.

## Uninstall

```bash
pip uninstall askrex-assistant
```

Then remove local runtime state only if you no longer need it:

```bash
rm -rf .venv data logs transcripts Memory
```
