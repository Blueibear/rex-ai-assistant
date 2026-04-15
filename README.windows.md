# AskRex Assistant Windows Quickstart

This guide is for native Windows 10/11 setup. AskRex currently requires Python 3.11; do not use a system `python` that points at 3.12, 3.13, or 3.14.

## Prerequisites

- Python 3.11 from python.org or the Windows Store, with the `py` launcher available
- Git
- FFmpeg on `PATH`
- Microphone and speakers for voice mode
- Node.js/npm if you want to run the Electron desktop GUI
- Optional NVIDIA GPU for CUDA installs

## Install

```powershell
git clone https://github.com/Blueibear/AskRex-Assistant.git
cd AskRex-Assistant

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
pip install .

Copy-Item config\rex_config.example.json config\rex_config.json -ErrorAction SilentlyContinue
Copy-Item .env.example .env -ErrorAction SilentlyContinue
```

For voice mode, install the ML/audio stack:

```powershell
pip install -r requirements-cpu.txt
```

For the validated CUDA 12.4 stack:

```powershell
pip install -r requirements-gpu-cu124.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Configure

Edit:

- `config\rex_config.json` for runtime settings such as LLM provider, model, wake word, audio devices, integrations, and UI options.
- `.env` for secrets such as `OPENAI_API_KEY`, `HA_TOKEN`, `REX_SPEAK_API_KEY`, `REX_TOOL_API_KEY`, Twilio credentials, and search/weather keys.

Useful checks:

```powershell
rex-config show
python -m rex doctor
```

## Run

Text chat:

```powershell
python -m rex
```

Voice loop:

```powershell
python rex_loop.py
```

Python web dashboard:

```powershell
rex-gui
```

Electron desktop GUI:

```powershell
cd gui
npm.cmd install
npm.cmd run dev
```

Build/preview Electron:

```powershell
cd gui
npm.cmd run typecheck
npm.cmd run build
npm.cmd run preview
```

TTS API:

```powershell
$env:REX_SPEAK_API_KEY="example-key"  # pragma: allowlist secret
rex-speak-api
```

The TTS API binds to `http://127.0.0.1:5005` by default and accepts either `X-API-Key` or `Authorization: Bearer <key>`.

OpenClaw tool server:

```powershell
$env:REX_TOOL_API_KEY="example-key"  # pragma: allowlist secret
rex-tool-server
```

The tool server binds to `http://127.0.0.1:18790` by default.

## Notes

- `python run_gui.py` and `python gui.py` are deprecated Tkinter launchers. Use `rex-gui` or the Electron GUI.
- Runtime memory profiles live under `Memory\<user_id>\`; structured memory uses `data\memory\`; GUI chat history uses `data\history.db`.
- `simpleaudio` is intentionally optional on Windows because it has known build issues.
- `speexdsp_ns` is not required.
- Web search requires configured search credentials for Brave or SerpAPI; DuckDuckGo fallback behavior depends on the plugin path.
- Use `py -3.11 -m rex doctor` if your default `python` points at an unsupported version.

## Tests

```powershell
pip install -e ".[dev,test]"
pytest

cd gui
npm.cmd run typecheck
npm.cmd run build
```
