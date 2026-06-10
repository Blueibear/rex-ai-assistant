# AskRex Assistant Architecture

AskRex Assistant is a Python 3.11 local-first assistant with text chat, voice
interaction, memory, tool routing, workflow planning, integrations, and an
Electron desktop app under `gui/` as the current primary GUI. The Flask service
started by `rex-gui` remains a local API/runtime surface with an incomplete,
experimental browser dashboard at `/ui/`.

The PyPI/package name is `askrex-assistant`. The user-facing product name is
AskRex Assistant.

## Subsystems at a Glance

This section is the quickest way for a new contributor to orient. Each
subsystem gets a short summary and pointers to the most relevant files or
folders.

### Voice loop

The voice loop ties wake-word detection, speech-to-text, the LLM reply, and
text-to-speech into one runnable pipeline. The runnable entry script is
`rex_loop.py`, which imports `build_voice_loop` from the canonical package
implementation at `rex/voice_loop.py` (with an optimized variant in
`rex/voice_loop_optimized.py`).

### Speech-to-text (STT)

STT uses OpenAI Whisper, loaded lazily inside the voice loop so the heavy
model only initializes when voice mode is actually used. The transcription
helpers and Whisper integration live in `rex/voice_loop.py`; device
selection (`cuda` vs `cpu`) is resolved at model-load time from
`AppConfig.whisper_device`.

### LLM response handling

The `Assistant` class in `rex/assistant.py` is the single entry point for
generating replies. It injects system context, routes through the tool
layer, and delegates raw model calls to the providers in
`rex/llm_client.py` (local Transformers, OpenAI-compatible, Anthropic, or
Ollama). The voice loop must always call `Assistant.generate_reply()`
rather than the LLM client directly so tool routing is preserved.

### Text-to-speech (TTS)

TTS is served by a small Flask service in `rex_speak_api.py`, exposed as the
`rex-speak-api` console script on `127.0.0.1:5005`. It supports Coqui XTTS
voice cloning and optional `edge-tts` / `pyttsx3` fallbacks, and requires
`REX_SPEAK_API_KEY` for authenticated `POST /speak` requests.

### Wake word

Wake-word detection is owned by the `rex/wakeword/` package, with
`rex/wakeword/listener.py` as the listener and `rex/wakeword/utils.py` as
the shared helpers. The default backend is openWakeWord; `custom_embedding`
and `custom_onnx` paths look for assets under `config/wake_words/hey_rex/`.

### Home Assistant integration

The high-level Home Assistant client and helpers live in
`rex/ha_bridge.py`, with the supporting device/discovery/state code under
`rex/ha/`. The Electron Home Assistant page and `/api/ha/test` endpoint use
this layer to list entities and verify connectivity, and `rex ha ...` CLI
commands are wired through `rex/cli.py`.

### Desktop GUI

The current primary GUI is the Electron + React app in `gui/`, built with
Vite and packaged from `gui/dist-electron/`. The Python/Flask local API and
the experimental browser dashboard at `/ui/` live in `rex/gui_app.py`
(launched via the `rex-gui` console script); the legacy Tkinter
`gui.py` / `run_gui.py` entry points are **archived** (moved to `archived/tkinter_gui/`).

### Plugin / tool system

Built-in tools are registered and executed by the OpenClaw-facing layer at
`rex/openclaw/`: `tool_registry.py` builds the registry, `tool_executor.py`
enforces policy and runs the tools, individual adapters live under
`rex/openclaw/tools/`, and `tool_server.py` (`rex-tool-server`) exposes them
over HTTP. Optional legacy plugins (for example `web_search`) and example
skills sit under `plugins/` and `plugins/skills/`.

### Configuration files

Runtime, non-secret settings live in `config/rex_config.json`, loaded and
validated by `rex/config.py` (`AppConfig`). Secrets such as
`OPENAI_API_KEY`, `HA_TOKEN`, `REX_SPEAK_API_KEY`, `REX_TOOL_API_KEY`, and
`OPENCLAW_GATEWAY_TOKEN` live in `.env`, and per-user profile overrides live
in `profiles/<name>.json`.

## Runtime Shape

| Layer | Main modules | Notes |
|---|---|---|
| CLI | `rex/__main__.py`, `rex/cli.py` | `python -m rex` and console script `rex` |
| Core assistant | `rex/assistant.py`, `rex/llm_client.py` | LLM selection, tool-aware replies, system context |
| Voice loop | `rex_loop.py`, `rex/voice_loop.py`, `rex/voice_loop_optimized.py` | Wake word, STT, LLM, and TTS path |
| Config | `rex/config.py`, `rex/config_manager.py`, `config/rex_config.json` | Runtime JSON config plus `.env` secrets |
| Memory and history | `rex/memory.py`, `rex/memory_utils.py`, `rex/history_store.py`, `Memory/`, `data/` | Per-user memory plus command/chat history |
| Tools | `rex/openclaw/tool_registry.py`, `rex/openclaw/tool_executor.py`, `rex/openclaw/tools/` | Local tool registry and executor |
| Tool server | `rex/openclaw/tool_server.py` | `rex-tool-server` on `127.0.0.1:18790` — **developer-only** |
| Electron UI | `gui/` | Current primary React/Electron GUI, built to `gui/dist-electron/` — **shippable** |
| Python/Flask API and experimental web UI | `rex/gui_app.py` | `rex-gui` on `127.0.0.1:8765`; local APIs plus incomplete `/ui/` browser dashboard — **developer-only** |
| TTS API | `rex_speak_api.py` | `rex-speak-api` on `127.0.0.1:5005` — **developer-only** |
| Computer agent | `rex/computers/agent_server.py` | `rex-agent`, local agent API for controlled OS automation — **developer-only** |

## Repository Layout

```text
.
|-- rex/                     # Main Python package
|   |-- cli.py               # CLI command tree
|   |-- assistant.py         # Assistant orchestration
|   |-- config.py            # AppConfig loader and rex-config CLI
|   |-- llm_client.py        # Transformers/OpenAI/Anthropic/Ollama clients
|   |-- voice_loop.py        # Package voice-loop exports
|   |-- voice_loop_optimized.py
|   |-- gui_app.py           # Flask local API plus experimental browser dashboard
|   |-- openclaw/            # Tool registry, executor, bridges, tool server
|   |-- integrations/        # Email, calendar, SMS service adapters
|   |-- computers/           # Remote computer agent/client support
|   |-- wakeword/            # Wake-word helpers
|   |-- notifications/       # Newer notification package pieces
|   `-- ...                  # Scheduler, workflows, memory, auth, etc.
|-- gui/                     # Electron + React desktop app
|-- plugins/                 # Optional legacy plugin modules, e.g. web_search
|-- config/                  # Runtime JSON config examples/defaults
|-- Memory/                  # Per-user profile and memory data
|-- data/                    # Local SQLite/state files at runtime
|-- tests/                   # Pytest suite
|-- docs/                    # Current docs plus archived planning/history
|-- rex_loop.py              # Voice loop runner
|-- rex_speak_api.py         # TTS Flask API
|-- flask_proxy.py           # Deprecated legacy compatibility proxy (see SURFACE-CLASSIFICATION.md)
`-- pyproject.toml           # Package metadata and console scripts
```

Top-level modules such as `config.py`, `llm_client.py`, and `memory_utils.py`
remain compatibility shims for older imports. New code should import from
`rex.*`.

## Entry Points

Defined in `pyproject.toml`:

| Console script | Target | Classification |
|---|---|---|
| `rex` | `rex.cli:main` | shippable |
| `rex-config` | `rex.config:cli` | developer-only |
| `rex-speak-api` | `rex_speak_api:main` | developer-only |
| `rex-agent` | `rex.computers.agent_server:main` | developer-only |
| `rex-gui` | `rex.gui_app:main` | developer-only |
| `rex-tool-server` | `rex.openclaw.tool_server:main` | developer-only |

Module/script entry points:

| Command | Purpose |
|---|---|
| `python -m rex` | Default CLI chat |
| `python -m rex doctor` | Environment diagnostics |
| `python rex_loop.py` | Full local voice loop |
| `python rex_speak_api.py` | Equivalent TTS API script form |
| `python flask_proxy.py` | **Deprecated** legacy compatibility proxy/API — use `rex-gui` instead |

`python -m rex-speak-api` is not a valid module invocation; use
`rex-speak-api` or `python rex_speak_api.py`.

## CLI Command Tree

`rex --help` currently exposes commands for:

- diagnostics: `doctor`, `version`, `tools`, `usage`
- chat and memory: `chat`, `memory`, `remember`, `history`, `quick-actions`
- knowledge: `kb`
- workflows: `plan`, `run-workflow`, `workflows`, `executor`, `approvals`
- schedule and reminders: `scheduler`, `reminders`, `cues`
- communications: `email`, `calendar`, `msg`, `notify`
- automation: `browser`, `os`, `gh`, `code`, `pc`
- integrations: `ha`, `wp`, `wc`
- identity and shopping: `whoami`, `identify`, `voice-id`, `shopping`

## Configuration Model

AskRex uses a split configuration model:

- `.env` stores secrets and service-specific environment controls.
- `config/rex_config.json` stores runtime settings such as wake word, models,
  audio devices, integrations, workflows, and UI defaults.

The canonical wake-word section is `wakeword`. The legacy `wake_word` key is
migrated at runtime with a warning.

`rex-config migrate-legacy-env` migrates older non-secret environment variables
into `config/rex_config.json` without overwriting non-default runtime values.

## Tool Architecture

Local tool execution is owned by `rex/openclaw/`:

- `tool_registry.py` builds the local registry.
- `tool_executor.py` enforces policy and executes tools.
- `tools/` contains individual tool adapters.
- `tool_server.py` exposes tools over HTTP for OpenClaw-compatible callers.

The tool server listens on `127.0.0.1:18790` by default and requires
`REX_TOOL_API_KEY` for `/rex/tools/{tool_name}` calls.

The built-in tool set includes time, weather, web search, email, SMS, calendar,
Home Assistant, Plex, WordPress, and WooCommerce paths. Optional integrations
only become usable when their dependencies and credentials are configured.

## UI Architecture

### Electron Desktop App

The Electron app lives in `gui/` and uses Electron/Vite/React. It is the current
primary user-facing GUI. Its package scripts are:

```bash
npm.cmd run dev
npm.cmd run typecheck
npm.cmd run build
npm.cmd run preview
npm.cmd run lint
```

The current Electron routes include home, devices, chat, voice, tasks,
calendar, reminders, memories, email, SMS, notifications, shopping, logs,
history, usage, integrations, settings, Home Assistant, quick actions, and
about.

For Electron-only verification harnesses, build first so
`gui/dist-electron/main/index.js` matches TypeScript sources.

### Python/Flask API and Experimental Browser Dashboard

`rex-gui` starts `rex/gui_app.py`, serves local JSON/SSE endpoints, and also
serves an incomplete browser dashboard at `/ui/`. Treat the Flask surface as
backend/API and compatibility infrastructure, not the current primary GUI.
Representative endpoints include:

- `/api/dashboard/status`
- `/api/chat/send`
- `/api/logs/stream`
- `/api/usage`
- `/api/devices`
- `/api/ha/test`
- `/api/quick-actions`
- `/api/status/stream`
- `/api/history`
- `/api/integrations`
- `/api/calendar/events`
- `/api/email/inbox`
- `/api/sms/threads`
- `/api/tools`

### Legacy UI Surfaces

`gui.py` and `run_gui.py` are **archived** Tkinter-era entry points (moved to `archived/tkinter_gui/`). `flask_proxy.py`
is a **deprecated** compatibility API/proxy surface — use `rex-gui` instead.

## TTS API

`rex-speak-api` runs `rex_speak_api.py` on `127.0.0.1:5005` by default. It
requires `REX_SPEAK_API_KEY` and accepts the key through `X-API-Key` or
`Authorization: Bearer ...`.

Main endpoints:

- `GET /health/live`
- `GET /health/ready`
- `POST /speak`

The service can optionally register Home Assistant and shopping blueprints when
their imports/configuration are available.

## OpenClaw Integration

AskRex integrates with OpenClaw over HTTP rather than importing OpenClaw as a
Python package. Gateway settings live under the `openclaw` key in
`config/rex_config.json`; the gateway secret is `OPENCLAW_GATEWAY_TOKEN` in
`.env`.

Primary modules:

- `rex/openclaw/http_client.py`
- `rex/openclaw/tool_bridge.py`
- `rex/openclaw/event_bridge.py`
- `rex/openclaw/browser_bridge.py`
- `rex/openclaw/voice_bridge.py`
- `rex/openclaw/tool_server.py`

## Testing and Quality

Pytest configuration is in `pyproject.toml`. Common checks:

```bash
pytest -q
python -m rex --help
python -m rex doctor
python scripts/security_audit.py
```

Electron checks:

```powershell
cd gui
npm.cmd run typecheck
npm.cmd run build
```

Coverage is configured in `pyproject.toml` with a `fail_under` threshold of 75.

## Design Rules

- Put new core code under `rex/`.
- Keep top-level Python files as compatibility shims or explicit entry scripts.
- Keep secrets out of `config/rex_config.json`.
- Make optional integrations fail closed or degrade gracefully when unconfigured.
- Bind network services to localhost unless a deployment explicitly opts out.
- Update docs when console scripts, ports, config keys, or UI surfaces change.
