# Advanced / Developer Install

This document covers install paths beyond the shortest quick start: editable
development installs, optional ML/audio stacks, GPU wheels, Docker, and the
Electron desktop app.

Current compatibility policy:

- Supported Python version: `>=3.11,<3.12`
- Best validated Windows path: Python 3.11 plus `requirements-gpu-cu124.txt`
- Python 3.12+ is not currently supported by the full ML/TTS dependency stack

## Manual Install: macOS / Linux

```bash
git clone https://github.com/Blueibear/AskRex-Assistant.git
cd AskRex-Assistant

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
cp .env.example .env

pip install -e .
pip install -r requirements-cpu.txt

python -m rex doctor
python -m rex
```

## Manual Install: Windows PowerShell

```powershell
git clone https://github.com/Blueibear/AskRex-Assistant.git
cd AskRex-Assistant

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
Copy-Item .env.example .env

pip install -e .
pip install -r requirements-cpu.txt

python -m rex doctor
python -m rex
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Optional Extras

Use extras for development and optional non-CUDA dependency groups:

```bash
pip install -e ".[dev]"
pip install -e ".[test]"
pip install -e ".[ml,audio]"
pip install -e ".[full]"
pip install -e ".[voice-id]"
```

Use the split requirements files for PyTorch GPU installs. CUDA wheels need the
PyTorch wheel index, so GPU extras are intentionally not used.

## CPU and GPU Requirements

CPU-only:

```bash
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-cpu.txt
```

CUDA 12.4:

```bash
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-gpu-cu124.txt
```

CUDA 11.8:

```bash
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-gpu.txt
```

Verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Interactive Installer

The repo still includes `install.py` for supervised local setup:

```bash
python install.py
python install.py --with-ml
python install.py --with-dev
python install.py --auto-install-ffmpeg
python install.py --mic-test
```

Prefer the manual install commands above when you need exact dependency control.

## Electron Desktop App

The desktop app lives under `gui/`.

```powershell
cd gui
npm.cmd install
npm.cmd run dev
```

Verification/build:

```powershell
cd gui
npm.cmd run typecheck
npm.cmd run build
npm.cmd run preview
```

For Electron-only verification harnesses, run `npm.cmd run build` before a
`gui/tmp_verify_*.cjs` harness requires `gui/dist-electron/main/index.js`.

## Python/Flask API and Experimental Web Dashboard

```bash
rex-gui
```

Open:

```text
http://127.0.0.1:8765/ui/
```

Override the port with `REX_GUI_PORT`.

`rex-gui` is useful for local Flask API routes and dashboard compatibility
checks. The browser dashboard at `/ui/` is incomplete in current testing; use
the Electron app as the primary GUI.

## Optional Local Services

TTS API:

```bash
REX_SPEAK_API_KEY=replace-with-random-secret rex-speak-api
```

Default address:

```text
http://127.0.0.1:5005
```

Tool server:

```bash
REX_TOOL_API_KEY=replace-with-random-secret rex-tool-server
```

Default address:

```text
http://127.0.0.1:18790
```

Voice loop:

```bash
python rex_loop.py
```

## Docker

The Dockerfile uses Python 3.11 and CPU PyTorch by default. The image default
command is `python -m rex`.

```bash
docker build -t askrex-assistant .
docker run --rm --env-file .env -it askrex-assistant
```

TTS API container example:

```bash
docker run --rm --env-file .env -p 5005:5005 \
  -it askrex-assistant rex-speak-api
```

Mount runtime state when you need persistence:

```bash
docker run --rm --env-file .env \
  -v "$(pwd)/Memory:/app/Memory" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/transcripts:/app/transcripts" \
  -it askrex-assistant
```

## Development Checks

```bash
pytest -q
python -m rex --help
python -m rex doctor
python scripts/security_audit.py
```

Format/lint:

```bash
ruff check .
black --check .
mypy .
```

Electron:

```powershell
cd gui
npm.cmd run typecheck
npm.cmd run build
```

## Configuration Reminder

- `.env` is for secrets and service-specific environment controls.
- `config/rex_config.json` is for runtime settings.
- Use `rex-config show` to inspect resolved config.
- Use `rex-config migrate-legacy-env` for old non-secret environment variables.
