# AskRex Assistant Instruction Manual

This manual is the current single-document guide for installing, configuring,
running, and troubleshooting AskRex Assistant.

## 1. What AskRex Is

AskRex Assistant is a local-first AI assistant with:

- CLI text chat
- wake-word voice interaction
- Whisper-based speech-to-text
- LLM backends for local Transformers, OpenAI, Anthropic, and Ollama
- text-to-speech backends such as XTTS, Edge TTS, Piper, and pyttsx3 paths
- memory, knowledge base, scheduler, reminders, notifications, workflows, and tools
- integrations for Home Assistant, email, calendar, SMS, GitHub, browser/OS automation, WordPress, WooCommerce, and remote computer control
- Python web dashboard and Electron desktop app

## 2. Supported Runtime

- Python: `>=3.11,<3.12`
- Primary package name: `askrex-assistant`
- Main CLI: `rex` or `python -m rex`
- Best documented Windows path: Python 3.11 in PowerShell

Do not use Python 3.12+ for a full install unless the dependency stack has been
validated and the package metadata has changed.

## 3. Configuration Model

AskRex uses a split configuration model:

- `.env` stores secrets and service-specific environment controls.
- `config/rex_config.json` stores runtime settings.

Secrets include API keys, tokens, and passwords. Runtime settings include wake
word, audio devices, model/provider selection, integration config, and workflow
defaults.

The canonical wake-word section is `wakeword`. The legacy `wake_word` key is
still migrated at runtime with a warning.

Useful config commands:

```bash
rex-config show
rex-config migrate-legacy-env --dry-run
rex-config migrate-legacy-env
```

## 4. Install

Windows PowerShell:

```powershell
git clone https://github.com/Blueibear/AskRex-Assistant.git
cd AskRex-Assistant
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e .
Copy-Item .env.example .env
```

macOS/Linux:

```bash
git clone https://github.com/Blueibear/AskRex-Assistant.git
cd AskRex-Assistant
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
cp .env.example .env
```

Optional stacks:

```bash
pip install -r requirements-cpu.txt
pip install -r requirements-gpu-cu124.txt
pip install -r requirements-gpu.txt
pip install -r requirements-dev.txt
pip install -e ".[ml,audio]"
pip install -e ".[full]"
```

Use requirements files for GPU PyTorch installs; do not use old GPU extras.

## 5. First Checks

```bash
python -m rex --help
python -m rex doctor
python -m rex tools --all
```

Start with text mode before adding voice, web, Electron, or services:

```bash
python -m rex
```

## 6. Main Runtime Surfaces

| Surface | Command | Default |
|---|---|---:|
| CLI chat | `python -m rex` | none |
| Voice loop | `python rex_loop.py` | none |
| Python web dashboard | `rex-gui` | `http://127.0.0.1:8765/ui/` |
| Electron desktop | `npm.cmd run dev` in `gui/` | app window |
| TTS API | `rex-speak-api` | `http://127.0.0.1:5005` |
| Tool server | `rex-tool-server` | `http://127.0.0.1:18790` |
| Computer agent | `rex-agent` | localhost agent API |
| Legacy proxy | `python flask_proxy.py` | `http://127.0.0.1:5000` |

Use `rex-gui` or Electron for UI work. `gui.py` and `run_gui.py` are deprecated.
Treat `flask_proxy.py` as a compatibility surface, not the primary dashboard.

## 7. Voice Setup

List audio devices:

```bash
python audio_config.py --list
```

Run the voice loop:

```bash
python rex_loop.py
```

If detection fails:

```bash
python -m rex doctor
python audio_config.py --list
python wakeword_listener.py
```

Optional custom wake-word checks:

```bash
python scripts/validate_wakeword_model.py --backend custom_onnx --model-path models/wakewords/hey_rex.onnx
python scripts/validate_wakeword_model.py --backend custom_embedding --embedding-path models/wakewords/hey_rex.pt
```

## 8. Python Web Dashboard

```bash
rex-gui
```

Open:

```text
http://127.0.0.1:8765/ui/
```

Common API checks:

```bash
curl http://127.0.0.1:8765/api/dashboard/status
curl http://127.0.0.1:8765/api/tools
```

Override the port with `REX_GUI_PORT`.

## 9. Electron Desktop App

```powershell
cd gui
npm.cmd install
npm.cmd run dev
```

Build and verify:

```powershell
cd gui
npm.cmd run typecheck
npm.cmd run build
npm.cmd run preview
```

For Electron-only verification harnesses, run `npm.cmd run build` first and
place the harness under `gui/tmp_verify_*.cjs`. The harness should require
`gui/dist-electron/main/index.js`, wait for the main `BrowserWindow`, and drive
the renderer with `webContents.executeJavaScript()`.

## 10. TTS API

Set a secret and start:

```powershell
$env:REX_SPEAK_API_KEY = "replace-with-a-random-secret"
rex-speak-api
```

Smoke test:

```powershell
$headers = @{
  "Content-Type" = "application/json"
  "X-API-Key" = $env:REX_SPEAK_API_KEY
}
$body = @{ text = "Hello from Rex"; user = "default" } | ConvertTo-Json
Invoke-WebRequest -Uri http://127.0.0.1:5005/speak -Method POST -Headers $headers -Body $body -OutFile speech.wav
```

Health:

```bash
curl http://127.0.0.1:5005/health/live
curl http://127.0.0.1:5005/health/ready
```

## 11. Tool Server

Set a secret and start:

```powershell
$env:REX_TOOL_API_KEY = "replace-with-a-random-secret"
rex-tool-server
```

Smoke test:

```powershell
$headers = @{
  "Content-Type" = "application/json"
  "Authorization" = "Bearer $env:REX_TOOL_API_KEY"
}
Invoke-WebRequest -Uri http://127.0.0.1:18790/rex/tools/time_now -Method POST -Headers $headers -Body "{}"
```

List local tool readiness:

```bash
python -m rex tools --all
```

## 12. Common CLI Commands

```bash
rex memory recent 5
rex remember "The garage code is 1234"
rex kb ingest notes.txt --title "Notes"
rex kb search "project plan"
rex scheduler list
rex reminders add "Call Sam" --when "tomorrow 9am"
rex email unread --limit 5
rex calendar upcoming --days 14
rex msg send --to "+15551234567" --body "Hello"
rex notify send --priority normal --title "Done" --body "Task completed"
rex shopping add milk
rex history --limit 20
rex quick-actions list
rex gh repos
rex wc orders list --site myshop
rex ha tts test --message "Hello from Rex"
```

## 13. Security Rules

- Never commit `.env`.
- Keep secrets out of `config/rex_config.json`.
- Bind services to localhost unless you intentionally deploy behind HTTPS and auth.
- Set `REX_SPEAK_API_KEY` for TTS API.
- Set `REX_TOOL_API_KEY` for the tool server.
- Use strict tokens and allowlists for `rex-agent`.
- Treat web content, email, SMS, tool results, and external URLs as untrusted.

## 14. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Unsupported Python | Python 3.12+ or wrong venv | Recreate `.venv` with Python 3.11 |
| `ffmpeg` missing | System dependency absent | Install ffmpeg and confirm it is on `PATH` |
| `torch` missing | Optional ML stack not installed | Install `requirements-cpu.txt` or a GPU requirements file |
| Wake word not detected | Mic/device/threshold issue | Run `audio_config.py --list` and check `wakeword` config |
| TTS 401 | Wrong API key | Use `X-API-Key` matching `REX_SPEAK_API_KEY` |
| TTS 429 | Rate limit hit | Adjust `REX_SPEAK_RATE_LIMIT` or reduce callers |
| Tool server 401 | Missing bearer token | Set and send `REX_TOOL_API_KEY` |
| Electron behavior stale | Built files stale | Run `npm.cmd run build` in `gui/` |
| GUI port conflict | `8765` in use | Set `REX_GUI_PORT` |

## 15. Most Useful Docs

Read in this order:

1. `README.md`
2. `INSTALL.md`
3. `RUNNING.md`
4. `CONFIGURATION.md`
5. `docs/usage.md`
6. `docs/UI_SURFACES.md`
7. `docs/runbook.md`
8. `docs/ARCHITECTURE.md`
9. `docs/troubleshooting.md`
10. `README.windows.md`
