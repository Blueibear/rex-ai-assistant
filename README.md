# AskRex Assistant

<p align="center">
  <img src="assets/brand/primary-horizontal.png" alt="AskRex Assistant - local-first voice AI" width="400" />
</p>

<p align="center">
  <img src="https://github.com/Blueibear/AskRex-Assistant/actions/workflows/ci.yml/badge.svg" alt="CI status" />
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT" />
</p>

AskRex Assistant is a local-first AI assistant for text chat, wake-word voice interaction, desktop control surfaces, and credential-gated home/productivity integrations. It runs from the Python `rex` package, with optional local ML dependencies for Whisper, openWakeWord, and XTTS, plus optional cloud/local LLM routing through OpenAI-compatible providers and Ollama.

AskRex is alpha software. The CLI, voice loop, configuration system, Python web dashboard, Electron GUI shell, and several integrations are implemented; integration readiness varies by backend and credentials. See [docs/claude/INTEGRATIONS_STATUS.md](docs/claude/INTEGRATIONS_STATUS.md) for the current readiness snapshot.

## Table of Contents

- [Quick Start](#quick-start)
- [Main Entry Points](#main-entry-points)
- [Features](#features)
- [Requirements](#requirements)
- [Configuration](#configuration)

## Quick Start

Python 3.11 is required. Python 3.12 and newer are intentionally rejected by the current installers and runtime checks because the validated ML/TTS dependency path is Python 3.11-only.

1. Clone the repo and create the Python 3.11 environment.

```powershell
git clone https://github.com/Blueibear/AskRex-Assistant.git
cd AskRex-Assistant
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

```bash
git clone https://github.com/Blueibear/AskRex-Assistant.git
cd AskRex-Assistant
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

2. Copy the local configuration files.

```powershell
Copy-Item config\rex_config.example.json config\rex_config.json -ErrorAction SilentlyContinue
Copy-Item .env.example .env -ErrorAction SilentlyContinue
```

```bash
cp -n config/rex_config.example.json config/rex_config.json
cp -n .env.example .env
```

3. Install Rex and the optional ML/audio stack.

```powershell
pip install .
.\install.ps1
pip install -r requirements-cpu.txt
```

```bash
pip install .
bash install.sh
pip install -r requirements-cpu.txt
# or, for validated Windows CUDA 12.4:
pip install -r requirements-gpu-cu124.txt
```

4. Configure LM Studio for local model access at `localhost:1234`.

Download and install LM Studio, start the LM Studio server on `localhost:1234`, and load your preferred model.

5. Verify the install and run Rex.

```powershell
rex doctor
rex
```

```bash
rex doctor
rex
```

## Main Entry Points

| Surface | Command | Status |
|---|---|---|
| CLI text chat | `rex` or `python -m rex` | Primary |
| Diagnostics | `rex doctor` or `python -m rex doctor` | Primary |
| Voice loop | `python rex_loop.py` | Primary voice runtime |
| Python web dashboard | `rex-gui` | Primary browser UI, serves `http://127.0.0.1:8765/ui/` |
| Electron desktop app | `cd gui && npm.cmd run dev` | Primary desktop shell for local development |
| TTS API | `rex-speak-api` | Service, default `127.0.0.1:5005` |
| OpenClaw tool server | `rex-tool-server` | Service, default `127.0.0.1:18790` |
| Windows computer agent | `rex-agent` | Optional remote PC control agent |
| Runtime config CLI | `rex-config` | Config inspection and legacy env migration |

`rex-gui` is the canonical GUI for the Rex AI Assistant. The legacy Tkinter launchers (`gui.py` and its entry point) are deprecated. Use `rex-gui` for the Python-served web UI or the Electron app under `gui/`.

## Features

| Area | Current repo state |
|---|---|
| Text chat | CLI chat uses `rex.assistant.Assistant` and the configured LLM provider. |
| Voice pipeline | Wake word via openWakeWord, STT via Whisper, LLM reply generation, and TTS via XTTS, edge-tts, or pyttsx3. |
| LLM providers | Local Transformers, OpenAI-compatible API settings, and Ollama routing are supported by config. |
| Configuration | Runtime settings live in `config/rex_config.json`; secrets live in `.env`; profiles live in `profiles/`. |
| GUIs | Python web dashboard (`rex-gui`) and Electron/React desktop GUI (`gui/`) both exist. Electron uses Python bridge scripts such as `rex_chat_stream_bridge.py`, `rex_tasks_bridge.py`, and `rex_voice_bridge.py`. |
| CLI integrations | `rex email`, `rex calendar`, `rex msg`, `rex notify`, `rex gh`, `rex code`, `rex pc`, `rex wp`, `rex wc`, `rex ha`, `rex shopping`, `rex usage`, and more are registered. |
| Tool execution | Tool registry, policy checks, audit logging, and OpenClaw-facing HTTP tool server are implemented. Local tool execution currently covers time, weather, and web search; the HTTP tool server exposes a broader adapter set. |
| Memory | Per-user profile data lives under `Memory/<user_id>/`; structured working/long-term memory lives under `data/memory/`; GUI chat history uses `data/history.db`. |
| Notifications | Priority routing, quiet hours, digest/escalation logic, CLI commands, and Electron notification UI plumbing are present; legacy Flask dashboard notification routes are not the current surface. |
| WordPress/WooCommerce | WordPress health checks and WooCommerce order/product reads are implemented; WooCommerce order status and coupon writes are approval-gated. |
| OpenClaw | HTTP gateway/client adapters and a standalone Rex tool server are present; feature flags under `openclaw` control gateway-backed paths. |

## Requirements

| Component | Requirement |
|---|---|
| Python | 3.11 only |
| OS | Windows 10/11, macOS, or Linux |
| FFmpeg | Required for parts of the audio/TTS stack |
| Audio hardware | Microphone and speakers for voice mode |
| Node.js/npm | Required only for the Electron GUI under `gui/` |
| GPU | Optional NVIDIA CUDA path via `requirements-gpu*.txt` |

On Windows, use `py -3.11 ...` or activate the repo `.venv` before running commands. A system default `python` that points at 3.12+ will be rejected.

## Configuration

AskRex uses three configuration layers:

1. `config/rex_config.json` for non-secret runtime settings such as audio, wake word, models, UI, integrations, and feature flags.
2. `.env` for secrets such as `OPENAI_API_KEY`, `HA_TOKEN`, `REX_SPEAK_API_KEY`, `REX_TOOL_API_KEY`, Twilio credentials, and search/weather keys.
3. `profiles/<name>.json` for profile-level capabilities and runtime overrides.

The canonical wake word config key is `wakeword`. The legacy `wake_word` key is still migrated at runtime but logs a warning.

Useful config commands:

```bash
rex-config show
rex-config migrate-legacy-env --dry-run
rex-config migrate-legacy-env
```

See [CONFIGURATION.md](CONFIGURATION.md), [docs/configuration.md](docs/configuration.md), and [docs/environment-variables.md](docs/environment-variables.md).

## GUI Usage

Python web dashboard:

```bash
rex-gui
# opens http://127.0.0.1:8765/ui/
```

Electron desktop GUI:

```powershell
cd gui
npm.cmd install
npm.cmd run dev

# Build and preview the compiled Electron app:
npm.cmd run build
npm.cmd run preview
```

The Electron app requires the Python bridge scripts at the repo root and the current `gui/dist-electron` build for built-app verification harnesses. See [docs/UI_SURFACES.md](docs/UI_SURFACES.md) and [docs/e2e-gui-launch-test.md](docs/e2e-gui-launch-test.md).

## Common Commands

```bash
rex --help
rex doctor
rex tools -v
rex usage
rex memory stats
rex kb search "query"
rex plan "check weather in Dallas"
rex approvals
rex wc orders list --site myshop
rex wc coupons create --site myshop --code SAVE10 --amount 10 --type percent
rex ha tts test --message "Hello from Rex"
```

## Services

TTS API:

```bash
set REX_SPEAK_API_KEY=change-me
rex-speak-api
```

Request format:

```bash
curl -X POST http://127.0.0.1:5005/speak \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" \
  -d '{"text":"Hello from AskRex","user":"default"}' \
  --output speech.wav
```

OpenClaw/Rex tool server:

```bash
set REX_TOOL_API_KEY=change-me
rex-tool-server
```

Health endpoints:

```bash
curl http://127.0.0.1:5005/health/live
curl http://127.0.0.1:18790/health/live
```

## Development

```bash
pip install -e ".[dev,test]"
pytest
ruff check .
black --check .
mypy .
```

Electron checks:

```powershell
cd gui
npm.cmd run typecheck
npm.cmd run build
```

The current coverage threshold in `pyproject.toml` is 75 percent. Test markers include `unit`, `integration`, `slow`, `audio`, `gpu`, `network`, `asyncio`, `anyio`, and `smoke`.

## Documentation

Start with [docs/INDEX.md](docs/INDEX.md). High-value active docs:

- [INSTALL.md](INSTALL.md) - installation guide
- [RUNNING.md](RUNNING.md) - runtime commands
- [docs/usage.md](docs/usage.md) - user-facing usage guide
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - architecture overview
- [docs/UI_SURFACES.md](docs/UI_SURFACES.md) - current UI surface inventory
- [docs/tools.md](docs/tools.md) - tool registry and execution notes
- [docs/api.md](docs/api.md) - HTTP services reference
- [docs/troubleshooting.md](docs/troubleshooting.md) - common failures

Files under `docs/archive/` are historical development records and may intentionally describe old plans or superseded states.

## Security

- Keep secrets in `.env` or `config/credentials.json`; do not put them in `config/rex_config.json`.
- `rex-speak-api` requires `REX_SPEAK_API_KEY`.
- `rex-tool-server` requires `REX_TOOL_API_KEY` for tool invocation.
- Approval-gated high-risk actions, such as WooCommerce writes and remote PC commands, require explicit confirmation flows.

Security references:

- [docs/security/SECURITY_ADVISORY.md](docs/security/SECURITY_ADVISORY.md)
- [docs/security/SECURITY_AUDIT_2026-01-08.md](docs/security/SECURITY_AUDIT_2026-01-08.md)
- [docs/security/VULNERABILITY-SCAN.md](docs/security/VULNERABILITY-SCAN.md)
- [docs/security/SECRET-SCAN.md](docs/security/SECRET-SCAN.md)

## Contributing

Use short-lived branches from `master`, open PRs back to `master`, and follow Conventional Commits. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Released under the MIT License. See [LICENSE](LICENSE).
