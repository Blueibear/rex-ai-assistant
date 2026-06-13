# AskRex Assistant

<p align="center">
  <img src="assets/brand/primary-horizontal.png" alt="AskRex Assistant - local-first voice AI" width="400" />
</p>

<p align="center">
  <img src="https://github.com/Blueibear/AskRex-Assistant/actions/workflows/ci.yml/badge.svg" alt="CI status" />
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT" />
  <a href="https://www.buymeacoffee.com/Blueibear" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 28px !important;width: 120px !important;" ></a>
</p>

AskRex Assistant is a local-first AI assistant for text chat, voice interaction, desktop control surfaces, and credential-gated home/productivity integrations. It runs from the Python `rex` package, with optional local ML dependencies for Whisper, openWakeWord, and XTTS, plus optional cloud/local LLM routing through OpenAI-compatible providers and Ollama.

AskRex is alpha software. It is useful for local testing and development, but it should not be treated as production-ready. Recent GUI and voice-loop fixes have made several paths usable end to end; wake-word tuning, warning cleanup, and per-user data separation are still in progress. See [docs/claude/INTEGRATIONS_STATUS.md](docs/claude/INTEGRATIONS_STATUS.md) for the broader integration readiness snapshot.

> **Advanced / Developer**: For CLI text mode, voice loop, GPU/CUDA setup, and backend service configuration, see the [Advanced / Developer](#advanced--developer) section below. For GPU/CUDA setup and additional install variants, see [docs/advanced-install.md](docs/advanced-install.md).

## Table of Contents

- [Getting Started](#getting-started)
- [Current Status](#current-status)
- [Working Now](#working-now)
- [Known Limitations / In Progress](#known-limitations--in-progress)
- [Planned Future Implementation](#planned-future-implementation)
- [Main Entry Points](#main-entry-points)
- [Features](#features)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [GUI Usage](#gui-usage)
- [Advanced / Developer](#advanced--developer)
- [Development](#development)
- [Documentation](#documentation)
- [Security](#security)

## Getting Started

The supported user-facing interface is the **Electron desktop app**. Python 3.11 and Node.js/npm are required.

Python 3.12 and newer are intentionally rejected by the current installers and runtime checks because the validated ML/TTS dependency path is Python 3.11-only.

1. Clone the repository.

   ```bash
   git clone https://github.com/Blueibear/AskRex-Assistant.git
   cd AskRex-Assistant
   ```

2. Run the install script for your platform.

   Windows PowerShell:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip setuptools wheel
   Copy-Item config\rex_config.example.json config\rex_config.json -ErrorAction SilentlyContinue
   Copy-Item .env.example .env -ErrorAction SilentlyContinue
   .\install.ps1
   ```

   macOS/Linux shell:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip setuptools wheel
   cp -n config/rex_config.example.json config/rex_config.json
   cp -n .env.example .env
   bash install.sh
   ```

3. Configure LM Studio for local chat completions.

   In LM Studio, load a chat model, start the local OpenAI-compatible server, and confirm the server URL is `http://localhost:1234/v1`. AskRex expects the LM Studio server to be reachable on `localhost:1234` when you choose that local provider path.

4. Run Rex in the Electron desktop app.

   Windows PowerShell:

   ```powershell
   cd gui
   npm.cmd install
   npm.cmd run dev
   ```

   macOS/Linux shell:

   ```bash
   cd gui
   npm install
   npm run dev
   ```

   The Electron app communicates with the Python Flask backend automatically.

5. Verify it works.

   ```bash
   rex doctor
   ```

   Then send a short chat message in the Electron app and confirm Rex replies.

## Current Status

This README reflects the current milestone after recent live testing and repair work. The Electron GUI is the current primary user-facing interface. The repo has usable CLI, chat, GUI, Home Assistant, and voice paths, with several areas still being stabilized.

## Working Now

- Core CLI help and doctor paths work: `rex --help`, `rex doctor`, and `python -m rex doctor`.
- Basic text chat works in the CLI and GUI.
- The Electron GUI launches and the main shell is stable.
- GUI pages load for Tasks, Reminders, Settings, Users, Integrations, Email, Calendar, and Home Assistant.
- The Home Assistant page loads and lists entities after the recent GUI/backend consistency fixes.
- The Home Assistant connection test now stays connected in the UI after a successful real connection test.
- GUI Chat shows a visible pending/thinking state while Rex is preparing a reply.
- Voice Hold to Talk records, transcribes, shows a pending state, gets a Rex reply, and can be used repeatedly.
- Wake-word mode is wired and can work end to end in live testing, but reliability and latency still need work.
- Long voice answers now use a cleaner spoken handoff that points the user to the on-screen transcript instead of reading long replies badly.
- Custom wake backends are wired: built-in openWakeWord fallback, `custom_embedding` as an interim path, and `custom_onnx` as the long-term target when a real asset is present.
- Day/date phrasing coverage in chat has improved for common variants.

## Known Limitations / In Progress

- Wake-word reliability is still inconsistent and may require threshold/device tuning.
- Wake-word response latency is slower than desired.
- The repo does not ship a real `Hey Rex` custom wake asset. A valid asset is still required for `custom_onnx` or `custom_embedding` to become active.
- Outlook integration status in the GUI should not be read as full mailbox/calendar sync readiness. Live Outlook email/calendar sync is still incomplete.
- CLI identity listing is currently polluted with test/demo users.
- A deprecated `wake_word` config warning still appears during startup.
- A `.env` permissions warning still appears.
- Some pages and integrations are still in active stabilization; do not assume every registered integration is production-ready.
- Per-user data isolation is planned but not yet complete.

## Planned Future Implementation

- Per-user GUI login.
- Personal vs shared data isolation.
- Personal vs shared integrations.
- Per-user chat histories with the ability to continue prior chats.
- Per-user reminders, tasks, notifications, and memories.
- Scoped document/context uploads with an upload-time choice for personal, shared, or universal access.
- Small GUI polish items, including increasing the size of the top-left AskRex logo.

## Main Entry Points

| Surface | Command | Status |
|---|---|---|
| **Electron desktop app** | Windows: `cd gui; npm.cmd run dev`; macOS/Linux: `cd gui && npm run dev` | **Primary user-facing interface** — current primary GUI under active stabilization |
| CLI text chat | `rex` or `python -m rex` | Working basic chat (developer / advanced) |
| Diagnostics | `rex doctor` or `python -m rex doctor` | Working |
| Voice loop | `python rex_loop.py` | Working in live testing; tuning ongoing (developer / advanced) |
| Python/Flask local API | `rex-gui` | Starts Flask on `http://127.0.0.1:8765`; backend service for Electron — not a standalone browser app (developer-only) |
| TTS API | `rex-speak-api` | Implemented service, default `127.0.0.1:5005` (developer / advanced) |
| OpenClaw tool server | `rex-tool-server` | Implemented service, default `127.0.0.1:18790` (developer / advanced) |
| Windows computer agent | `rex-agent` | Optional remote PC control agent (developer / advanced) |
| Runtime config CLI | `rex-config` | Config inspection and legacy env migration (developer / advanced) |

The Electron app under `gui/` is the current primary GUI. `rex-gui` remains useful as a local Flask/API service and for compatibility testing, but its browser dashboard at `/ui/` is incomplete and should not be treated as the main user interface. The legacy Tkinter launchers (`gui.py` and its entry point) are archived — moved to `archived/tkinter_gui/` and no longer maintained. See [SURFACE-CLASSIFICATION.md](SURFACE-CLASSIFICATION.md).

## Features

| Area | Current repo state |
|---|---|
| Text chat | Basic CLI and GUI chat work through `rex.assistant.Assistant` and the configured LLM provider. Common day/date phrasing coverage has improved; provider quality still depends on model/configuration. |
| Voice pipeline | Hold to Talk works. Wake-word mode is wired and can work end to end through openWakeWord, Whisper STT, LLM reply generation, and TTS, but reliability and latency are still being tuned. Long answers now hand off to the transcript instead of reading a long reply aloud badly. |
| Custom wake support | Built-in openWakeWord remains the safe fallback path. `custom_embedding` is usable as an interim path, and `custom_onnx` is the target path for a real `Hey Rex` wake model. The repo does not ship that custom asset by default. |
| LLM providers | Local Transformers, OpenAI-compatible API settings, and Ollama routing are supported by config. Local model output quality varies by model and prompt path. |
| Configuration | Runtime settings live in `config/rex_config.json`; secrets live in `.env`; profiles live in `profiles/`. |
| GUIs | The Electron/React desktop GUI (`gui/`) is the current primary interface. The Electron shell is stable in current testing; Tasks, Reminders, Settings, Users, Integrations, Email, Calendar, and Home Assistant pages load. The Python/Flask `rex-gui` surface still serves local API routes and an experimental `/ui/` browser dashboard, but that browser UI is incomplete and not recommended as the primary interface. |
| Home Assistant | The GUI Home Assistant page loads and lists entities after recent fixes. The connection test performs a real backend connectivity check and persists success in the UI. |
| Email and calendar integrations | GUI integration status is more honest than before, but Outlook-connected status should not be read as full live Outlook mailbox/calendar sync. Treat Outlook email/calendar paths as partial until end-to-end sync is working. |
| CLI integrations | `rex email`, `rex calendar`, `rex msg`, `rex notify`, `rex gh`, `rex code`, `rex pc`, `rex wp`, `rex wc`, `rex ha`, `rex shopping`, `rex usage`, and more are registered. Readiness varies by backend credentials and test coverage. |
| Tool execution | Tool registry, policy checks, audit logging, and OpenClaw-facing HTTP tool server are implemented. Local tool execution currently covers time, weather, and web search; the HTTP tool server exposes a broader adapter set. |
| Memory | Memory storage paths exist under `Memory/` and `data/memory/`, and GUI chat history uses `data/history.db`. Full per-user GUI history/memory isolation is planned, not complete. |
| Notifications | Priority routing, quiet hours, digest/escalation logic, CLI commands, and Electron notification UI plumbing are present. Full per-user notification behavior is planned, not complete. |
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

> **Note**: On Windows, audio playback requires simpleaudio. Install it with `pip install simpleaudio` if audio output is not working.

## Configuration

AskRex uses three configuration layers:

1. `config/rex_config.json` for non-secret runtime settings such as audio, wake word, models, UI, integrations, and feature flags.
2. `.env` for secrets such as `OPENAI_API_KEY`, `HA_TOKEN`, `REX_SPEAK_API_KEY`, `REX_TOOL_API_KEY`, Twilio credentials, and search/weather keys.
3. `profiles/<name>.json` for profile-level capabilities and runtime overrides.

The canonical wake word config key is `wakeword`. The legacy `wake_word` key is still migrated at runtime but logs a warning.

Custom wake asset defaults now follow:

- `config/wake_words/hey_rex/model.onnx` for `custom_onnx`
- `config/wake_words/hey_rex/embedding.pt` for `custom_embedding`

The repo wiring is in place for a real custom `Hey Rex` wake path, but a valid asset file is still required before that path becomes active.

Useful config commands:

```bash
rex-config show
rex-config migrate-legacy-env --dry-run
rex-config migrate-legacy-env
```

See [CONFIGURATION.md](CONFIGURATION.md), [docs/configuration.md](docs/configuration.md), and [docs/environment-variables.md](docs/environment-variables.md).

For Home Assistant, see [Home Assistant setup](docs/home-assistant-setup.md).

## GUI Usage

See [docs/usage.md](docs/usage.md) for the full usage guide including voice mode, autonomous workflows, and integrations.

### Electron desktop GUI - Windows setup

Use Windows PowerShell:

```powershell
cd gui
npm.cmd install
npm.cmd run dev

## Build and preview the compiled Electron app:
npm.cmd run build
npm.cmd run preview
```

### Electron desktop GUI - macOS setup

Use Terminal:

```bash
cd gui
npm install
npm run dev

npm run build
npm run preview
```

### Flask API backend for Electron (`rex-gui`)

`rex-gui` starts the Flask server that the Electron GUI calls. It is a **backend service**, not a
standalone browser app. Running it directly will log a warning that the Electron shell is not
detected.

```bash
rex-gui
## Flask API listens on http://127.0.0.1:8765/
## Not intended for direct browser use; open the Electron app instead
```

The Electron app requires the Python bridge scripts at the repo root and the current
`gui/dist-electron` build for built-app verification harnesses. See
[docs/UI_SURFACES.md](docs/UI_SURFACES.md) and [docs/e2e-gui-launch-test.md](docs/e2e-gui-launch-test.md).

The supported GUI interface is the **Electron desktop app**. Use `rex-gui` only as a backend API
server or for local API route testing.

## Advanced / Developer

The following runtime paths are for developers, advanced users, and contributors. They are not the primary user-facing path.

### CLI Text Mode

After setting up the Python environment (see [Getting Started](#getting-started) step 2), run:

```bash
rex doctor
rex
```

Configure LM Studio for local model access at `localhost:1234`, or configure an OpenAI-compatible provider in `config/rex_config.json`.

For GPU/CUDA setup and additional install variants, see [docs/advanced-install.md](docs/advanced-install.md). Additional install scripts (`install_full.sh`, `install_lean.sh`, `setup.sh`) are in `scripts/install/`.

### Voice Loop

```bash
python rex_loop.py
```

Wake word → STT → LLM → TTS. Wake-word reliability still requires tuning.

### Common Commands

After activating the virtual environment:

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

### TTS API

The TTS API (`rex-speak-api`) requires an API key. Generate one with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Windows PowerShell:

```powershell
$env:REX_SPEAK_API_KEY = "<YOUR_API_KEY>"
rex-speak-api
```

macOS/Linux shell:

```bash
export REX_SPEAK_API_KEY=<YOUR_API_KEY>
rex-speak-api
```

Request format with curl, using macOS/Linux shell syntax or Git Bash/WSL on Windows:

```bash
curl -X POST http://127.0.0.1:5005/speak \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <YOUR_API_KEY>" \
  -d '{"text":"Hello from AskRex","user":"default"}' \
  --output speech.wav
```

### OpenClaw / Rex Tool Server

The OpenClaw tool server (`rex-tool-server`) requires an API key. Generate one with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Windows PowerShell:

```powershell
$env:REX_TOOL_API_KEY = "<YOUR_API_KEY>"
rex-tool-server
```

macOS/Linux shell:

```bash
export REX_TOOL_API_KEY=<YOUR_API_KEY>
rex-tool-server
```

Health endpoints with curl, using macOS/Linux shell syntax or Git Bash/WSL on Windows:

```bash
curl http://127.0.0.1:5005/health/live
curl http://127.0.0.1:18790/health/live
```

## Docker

Run AskRex in a container. See [docs/docker.md](docs/docker.md) for build, run, and GPU options.

## Memory

AskRex stores per-user voice and conversation profiles under `Memory/`. See [docs/memory.md](docs/memory.md) for profile format and voice cloning notes.

## Troubleshooting

Common issues: missing ffmpeg, CUDA driver mismatches, wake word not triggering. See [docs/troubleshooting.md](docs/troubleshooting.md) for solutions.

## Development

Python checks, after activating the virtual environment on any platform:

```bash
pip install -e ".[dev,test]"
pytest
ruff check .
black --check .
mypy .
```

Electron checks on Windows PowerShell:

```powershell
cd gui
npm.cmd run typecheck
npm.cmd run build
```

Electron checks on macOS/Linux:

```bash
cd gui
npm run typecheck
npm run build
```

The current coverage threshold in `pyproject.toml` is 75 percent. Test markers include `unit`, `integration`, `slow`, `audio`, `gpu`, `network`, `asyncio`, `anyio`, and `smoke`.

Skipped-test inventory:

- [docs/testing/SKIPPED-TESTS-INVENTORY.md](docs/testing/SKIPPED-TESTS-INVENTORY.md) — classified inventory of all 140 skip markers (`optional-dep-skip`, `platform-skip`, `retired-surface-skip`, `temporary-bug-skip`)

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
