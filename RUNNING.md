# How to Run AskRex Assistant

Use an activated Python 3.11 environment for every Python command in this guide.

Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

## Core Runtime Modes

| Mode | Command | Purpose |
|---|---|---|
| Text chat | `rex` or `python -m rex` | Interactive CLI chat |
| Diagnostics | `rex doctor` or `python -m rex doctor` | Environment, dependency, and config checks |
| Voice loop | `python rex_loop.py` | Wake word, STT, LLM, and TTS pipeline |
| Python web dashboard | `rex-gui` | Browser UI at `http://127.0.0.1:8765/ui/` |
| Electron desktop GUI | `cd gui && npm.cmd run dev` | Desktop React/Electron shell |
| TTS API | `rex-speak-api` | Local speech synthesis service on port 5005 |
| OpenClaw tool server | `rex-tool-server` | HTTP tool adapter service on port 18790 |
| Windows computer agent | `rex-agent` | Optional remote PC control agent |

The old Tkinter launchers `python run_gui.py` and `python gui.py` are deprecated. They remain in the repo only for legacy debugging.

## Text Chat

```bash
rex
# or
python -m rex
```

Useful related commands:

```bash
rex --help
rex version --verbose
rex tools -v
rex usage
```

## Voice Loop

```bash
python rex_loop.py
```

Optional flags:

```bash
python rex_loop.py --user james
python rex_loop.py --enable-plugin web_search
```

Voice mode requires the ML/audio dependency stack:

```bash
pip install -r requirements-cpu.txt
# or, for validated Windows CUDA 12.4:
pip install -r requirements-gpu-cu124.txt
```

## Python Web Dashboard

```bash
rex-gui
```

By default this starts a local Flask app on `127.0.0.1:8765` and opens:

```text
http://127.0.0.1:8765/ui/
```

Override the port:

```bash
REX_GUI_PORT=9000 rex-gui
```

PowerShell:

```powershell
$env:REX_GUI_PORT=9000; rex-gui
```

## Electron Desktop GUI

Development:

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

Development Electron resolves canonical bridge scripts from `bridge/`. Packaged
Windows Electron resolves those scripts from `resources/bridge/` and uses only
its managed `resources/python/python.exe`; it does not use machine Python or a
source checkout.

Build the managed Voice installer from `gui/` with `npm.cmd run dist`. The
Voice profile includes CPU Whisper/Torch and bundled FFmpeg, so the installer
is substantially larger than the Core runtime. CUDA/GPU acceleration and XTTS
voice cloning remain optional external profiles rather than installer claims.

For Electron-only verification harnesses, run `npm.cmd run build` in `gui/` before using `gui/tmp_verify_*.cjs` so `gui/dist-electron/main/index.js` matches the TypeScript sources.

## TTS API

```bash
set REX_SPEAK_API_KEY=example-key  # pragma: allowlist secret
rex-speak-api
```

PowerShell:

```powershell
$env:REX_SPEAK_API_KEY="example-key  # pragma: allowlist secret"
rex-speak-api
```

Default URL:

```text
http://127.0.0.1:5005
```

Example request:

```bash
curl -X POST http://127.0.0.1:5005/speak \
  -H "Content-Type: application/json" \
  -H "X-API-Key: example-key  # pragma: allowlist secret" \
  -d '{"text":"Hello from AskRex","user":"default"}' \
  --output speech.wav
```

## OpenClaw Tool Server

```bash
set REX_TOOL_API_KEY=example-key  # pragma: allowlist secret
rex-tool-server
```

Default URL:

```text
http://127.0.0.1:18790
```

Health check:

```bash
curl http://127.0.0.1:18790/health/live
```

Example tool call:

```bash
curl -X POST http://127.0.0.1:18790/rex/tools/time_now \
  -H "Content-Type: application/json" \
  -H "X-API-Key: example-key  # pragma: allowlist secret" \
  -d '{"args":{"location":"Dallas, TX"},"context":{}}'
```

## Diagnostics

```bash
python -m rex doctor
```

The doctor command checks Python version, package importability, config files, API keys, audio devices, FFmpeg, LLM service reachability, wake word config, STT/TTS dependencies, and GPU availability.

## Common Problems

If `python -m rex` reports unsupported Python 3.12+, activate the `.venv` or call Python 3.11 explicitly:

```powershell
py -3.11 -m rex doctor
```

If wake-word or voice features fail, install the CPU/GPU ML requirements and confirm FFmpeg is available:

```bash
pip install -r requirements-cpu.txt
ffmpeg -version
python -m rex doctor
```

If `rex-gui` opens a placeholder page, build the web UI under `rex/ui/` or use the Electron GUI under `gui/`.
