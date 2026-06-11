# Claude Reference: Commands and Entry Points

Use this reference when a task touches CLI behavior, startup flows, package
scripts, service ports, or docs about commands.

## Python Runtime

- Supported Python: `>=3.11,<3.12`
- Use `py -3.11` on Windows when multiple Python versions are installed.
- Do not document Python 3.12+ as supported unless package metadata, tests, and
  install docs are updated together.

## Install

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

Optional stacks:

```bash
pip install -r requirements-cpu.txt
pip install -r requirements-gpu-cu124.txt
pip install -r requirements-dev.txt
pip install -e ".[ml,audio]"
```

## Console Scripts

| Script | Target | Purpose | Classification |
|---|---|---|---|
| `rex` | `rex.cli:main` | Main CLI | shippable |
| `rex-config` | `rex.config:cli` | Config inspection and legacy env migration | developer-only |
| `rex-speak-api` | `rex_speak_api:main` | TTS HTTP API on `127.0.0.1:5005` | developer-only |
| `rex-agent` | `rex.computers.agent_server:main` | Computer agent API | developer-only |
| `rex-gui` | `rex.gui_app:main` | Flask local API plus incomplete experimental `/ui/` browser dashboard on `127.0.0.1:8765` | developer-only |
| `rex-tool-server` | `rex.openclaw.tool_server:main` | Tool server on `127.0.0.1:18790` | developer-only |

`python -m rex-speak-api` is invalid. Use `rex-speak-api` or
`python rex_speak_api.py`.

## Main Commands

```bash
python -m rex
python -m rex --help
python -m rex doctor
python -m rex tools --all
python rex_loop.py
rex-gui
rex-speak-api
rex-tool-server
```

Representative `rex` subcommands:

- `doctor`, `chat`, `version`, `tools`, `usage`
- `memory`, `remember`, `kb`, `history`, `quick-actions`
- `plan`, `run-workflow`, `workflows`, `executor`, `approvals`
- `scheduler`, `reminders`, `cues`
- `email`, `calendar`, `msg`, `notify`
- `browser`, `os`, `gh`, `code`, `pc`
- `wp`, `wc`, `ha`, `voice-id`, `shopping`

## GUI Commands

Electron primary GUI:

```powershell
cd gui
npm.cmd install
npm.cmd run dev
npm.cmd run typecheck
npm.cmd run build
```

Python/Flask API and experimental browser dashboard:

```bash
rex-gui
```

Electron-only verification harnesses should build first, then require
`gui/dist-electron/main/index.js` from `gui/tmp_verify_*.cjs`.

## Service Ports

| Service | Default |
|---|---:|
| `rex-gui` | `127.0.0.1:8765` |
| `rex-speak-api` | `127.0.0.1:5005` |
| `rex-tool-server` | `127.0.0.1:18790` |
| deprecated legacy compatibility only: `flask_proxy.py` (see `SURFACE-CLASSIFICATION.md`) | `0.0.0.0:5000` |

Prefer localhost binding in docs unless a deployment explicitly configures
remote access and authentication.

## Assistant Pipeline Module Locations

`Assistant.generate_reply()` delegates to four component modules:

| Component | Module path |
|---|---|
| `ContextBuilder` | `rex/context/builder.py` |
| `IntentRouter` | `rex/intent/router.py` |
| `ActionDispatcher` | `rex/actions/dispatcher.py` |
| `ResponseBuilder` | `rex/response/builder.py` |

Helper extracted from `Assistant.__init__`:

- `rex.followup_engine.init_followup_engine(settings, user_id)` — returns `(engine, pending_prompt)`.
