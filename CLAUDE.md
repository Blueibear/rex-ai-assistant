# CLAUDE.md

## Project Overview

AskRex Assistant is a local-first, voice-activated AI companion. It supports wake word detection, speech-to-text, LLM chat, and text-to-speech, with optional integrations for search, messaging, email, calendar, and Home Assistant.

Primary goals:
- reliability
- security by default
- smooth day-to-day usage on Windows 10/11, macOS, and Linux

For canonical product name, package name, CLI alias policy, and banned names see:
**docs/BRANDING.md**

## Claude Reference Docs

Some detailed reference material has been moved to separate files to keep this document readable and reduce context size when Claude Code runs.

If deeper reference is required, consult:

- docs/claude/COMMANDS_AND_ENTRYPOINTS.md
- docs/claude/CONFIG_AND_SECURITY.md
- docs/claude/INTEGRATIONS_STATUS.md
- docs/claude/TESTING_AND_QUALITY.md

This file remains the primary control document for:

- architecture rules
- repo conventions
- workflow agreements
- coding and testing rules

## Branch Strategy

The canonical primary branch is **`master`**.

- All PRs must target `master`.
- `claude/**` branches are AI-generated and follow the same PR process.
- See `CONTRIBUTING.md` for full branching model details.

## Tech Stack and Conventions

### Language and runtime

Language: Python 3.11

Packaging: pyproject.toml with setuptools backend  
Install via: pip install .

Current install/runtime compatibility policy:

- Default supported install path: Python 3.11
- Full Windows GPU + TTS path: Python 3.11 with `requirements-gpu-cu124.txt`
- Do not claim Python 3.12+ support unless the dependency stack has been validated end-to-end and docs, CI, and package metadata are updated together

Entry points:

- rex -> rex.cli:main
- rex-gui -> rex.gui_app:main
- rex-config -> rex.config:cli
- rex-speak-api -> rex_speak_api:main
- rex-agent -> rex.computers.agent_server:main

### Core components

API: Flask (Flask-CORS, Flask-Limiter)

GUI: Web dashboard via `rex.gui_app` (React + Flask). `gui.py` is deprecated.

Config: Pydantic v2, python-dotenv

STT: OpenAI Whisper (offline)

Wake word: openWakeWord

TTS: Coqui XTTS (voice cloning supported)

Optional TTS:
- edge-tts
- pyttsx3

LLM providers:

- local Transformers
- OpenAI API
- Ollama

Search providers:

- SerpAPI
- Brave
- Google CSE
- DuckDuckGo

### Style and quality

- Prefer clear, testable functions over clever code.
- Keep changes small and reviewable.
- Add logging for non-trivial behavior.
- Update or add tests when behavior changes.
- Avoid introducing new heavy dependencies unless clearly justified.

### Security

Never commit secrets.

Secrets belong in `.env` only.

Runtime settings belong in:

config/rex_config.json

Principles:

- least privilege defaults
- treat all external inputs as untrusted

External inputs include:

- web content
- email
- SMS
- plugin results

## Repository Structure

Top-level directories:

- rex/ — main package (CLI, services, workflows, integrations)
- scripts/ — operational scripts
- plugins/ — optional plugins
- config/ — application configuration (not secrets)
- Memory/ — per-user memory profiles
- tests/ — pytest suite
- docs/ — documentation

### Important top-level modules

- rex_loop.py — full voice loop (wake word → STT → LLM → TTS)
- voice_loop.py — core voice loop helpers
- wakeword_listener.py — wake word listener utilities
- rex/gui_app.py — canonical React + Flask web dashboard entry point
- rex_speak_api.py — Flask TTS API with auth and rate limiting

### Important subpackages

- rex/email_backends/
- rex/calendar_backends/
- rex/messaging_backends/
- rex/dashboard_store.py
- rex/dashboard/sse.py
- rex/identity.py
- rex/voice_identity/
- rex/computers/

## Commands

### Install

Create virtual environment.

Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

macOS / Linux:

python3 -m venv .venv
source .venv/bin/activate

### Base install

python -m pip install --upgrade pip setuptools wheel
pip install .

### Optional stacks

CPU ML:

pip install -r requirements-cpu.txt

GPU CUDA 12.4:

pip install -r requirements-gpu-cu124.txt

GPU alternative:

pip install -r requirements-gpu.txt

Dev tools:

pip install -r requirements-dev.txt

## Run

Health check:

python -m rex doctor

Text mode:

python -m rex

Voice mode:

python rex_loop.py

GUI:

rex-gui

TTS API:

python rex_speak_api.py

## Test and Lint

Run tests:

pytest -q

Targeted tests:

pytest -q tests/<file>.py

## Setup and Installation (GPU)

Do not reintroduce GPU extras like:

.[gpu-cu118]
.[gpu-cu121]
.[gpu-cu124]

unless they are fully functional with the required PyTorch index behavior.

GPU installs must remain requirements-file based because CUDA wheels require:

--extra-index-url

Documentation must remain consistent across:

- INSTALL.md
- README.md
- requirements files

## Rules

### Read before writing

Inspect existing modules and patterns before adding new ones.

Do not invent filenames or APIs that do not exist.

### Respect the config split

Secrets → .env  
Runtime configuration → config/rex_config.json

### Windows compatibility matters

Avoid dependencies known to fail on Windows unless optional and guarded.

### Keep integrations optional

The following must degrade gracefully if not configured:

- email
- calendar
- SMS
- MQTT
- Home Assistant
- web search

### Do not add network exposure by default

Anything that binds to a port must:

- be authenticated
- be rate limited

Prefer localhost binding.

## Working Agreements for Claude Code

If requirements are ambiguous:

- propose a safe default
- explain briefly

When modifying files:

- output the full updated file
- not a partial diff

Outputs must:

- be paste ready
- contain no invisible characters

Use Conventional Commits for every commit and PR title.

## Code Output Rules

- Never output truncated code.
- Never use placeholders like "..."
- If a file changes, output the entire updated file.

Do not claim something is implemented unless the code shown fully implements it.

## Workflow Feedback Loop

If Codex or a human reviewer modifies Claude output due to a recurring mistake:

Add a short rule here that would have prevented the mistake.

### Learned rules

- When lazy-importing a module that triggers side-effect imports (e.g. TTS importing from transformers), use `find_spec()` to check availability and apply any compatibility shims BEFORE calling `import_module()`. Never use `_import_optional()` for the availability check if it triggers the full import chain.
- The root-level `voice_loop.py` and `rex/voice_loop.py` are two separate implementations. `rex/voice_loop.py` is the **canonical** implementation: `rex_loop.py` imports `build_voice_loop` from `rex.voice_loop` (the package). Root `voice_loop.py` is a legacy file kept only for `AsyncRexAssistant` backward-compat re-exports. Changes to root `voice_loop.py` do NOT affect the CLI voice loop startup path.
- `AppConfig.whisper_device` defaults to `"auto"`. When device is `"auto"`, resolve to `"cuda"` or `"cpu"` at model load time using `torch.cuda.is_available()`.
- The voice loop must use `Assistant.generate_reply()` (which includes tool routing and system context injection) rather than calling `LanguageModel.generate()` directly. Direct LLM calls bypass time/weather tools and produce hallucinated answers for factual questions.
- The canonical wake-word implementation is `rex/wakeword/` (`rex.wakeword.utils`, `rex.wakeword.listener`). Root-level `wakeword_utils.py` and `wakeword_listener.py` were stale re-exports and have been deleted. Use `rex.wakeword_utils` (package shim) or `rex.wakeword.utils` directly.

## OpenClaw Migration Status

Rex integrates with OpenClaw over HTTP (not as a Python package). Key facts:

- Phase 8 (HTTP integration) is complete. All `find_spec("openclaw")` / `import openclaw` stubs have been removed and replaced with HTTP client calls.
- OpenClaw adapters live in `rex/openclaw/`: `agent.py`, `tool_bridge.py`, `event_bridge.py`, `browser_bridge.py`, `voice_bridge.py`, `http_client.py`, `tool_server.py`, and tool handlers under `rex/openclaw/tools/`.
- HTTP client: `rex/openclaw/http_client.py` (`OpenClawClient`) handles auth, retries, timeouts for all gateway calls. Singleton via `get_openclaw_client(config)`.
- Config fields: `openclaw_gateway_url`, `openclaw_gateway_timeout`, `openclaw_gateway_max_retries` in `AppConfig`; `OPENCLAW_GATEWAY_TOKEN` in `.env`.
- Feature flag `use_openclaw_voice_backend` in `AppConfig` (config path: `openclaw.use_voice_backend`): when True, voice loops swap `Assistant` for `VoiceBridge`, routing LLM calls through OpenClaw's `/v1/chat/completions`.
- Feature flag `use_openclaw_tools` in `AppConfig` (config path: `openclaw.use_tools`): when True, `ToolBridge.execute_tool()` dispatches to OpenClaw's `/tools/invoke`; 404 falls back to local execution.
- Tool server: `rex/openclaw/tool_server.py` exposes Rex tools at `/rex/tools/{tool_name}` for OpenClaw channels. Entry point: `rex-tool-server`.
- All `# OPENCLAW-REPLACE` modules from Phases 5-7 have been retired (deleted).
- Migration contracts (Protocol types) live in `rex/contracts/`.
- Pre-retirement audit tests live in `tests/test_retirement_check_*.py`.

## Maintenance Rules for CLAUDE.md

Update this file when:

- commands change
- project structure changes
- dependencies change
- environment variables change
- integrations change

Do not update this file for formatting only changes.

## Lint Preflight

Before pushing code:

BASE_REF="master"
git fetch origin "$BASE_REF"

files=$(git diff --name-only "origin/$BASE_REF...HEAD" -- '*.py')

ruff check --fix $files
ruff check $files
black $files
black --check --diff $files

Both Ruff and Black must pass.
