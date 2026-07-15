# CLAUDE.md

## Project Overview

AskRex Assistant is a local-first, voice-activated AI companion. It supports wake word detection, speech-to-text, LLM chat, and text-to-speech, with optional integrations for search, messaging, email, calendar, and Home Assistant.

Primary goals:
- reliability
- security by default
- smooth day-to-day usage on Windows 10/11, macOS, and Linux

For canonical product name, package name, CLI alias policy, and banned names see:
**docs/BRANDING.md**

For the authoritative classification of every entry point and UI surface (shippable, developer-only, deprecated, archived, removed) see:
**SURFACE-CLASSIFICATION.md**

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

- rex -> rex.cli:main  # first-class: primary CLI
- rex-gui -> rex.gui_app:main  # backend service: Electron-backed GUI server; not a standalone browser app
- rex-config -> rex.config:cli  # utility: config inspection and migration
- rex-speak-api -> rex_speak_api:main  # backend service: TTS API with auth and rate limiting
- rex-agent -> rex.computers.agent_server:main  # backend service: optional remote PC control API
- rex-tool-server -> rex.openclaw.tool_server:main  # backend service: OpenClaw tool adapter at /rex/tools/{tool_name}

Mobile API gateway (issue #323) runs via the CLI, not a console script:

- python -m rex mobile-api [--host HOST] [--port PORT]  # authenticated mobile gateway (default 127.0.0.1:8765)
- python -m rex mobile-user create --username NAME      # safe mobile user creation (getpass prompts)

### Core components

API: Flask (Flask-CORS, Flask-Limiter, Flask-Sock for the mobile WebSocket)

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
- bridge/ — HTTP bridge modules for STT, TTS, calendar, and other external services
- archived/ — retired files kept for reference; not maintained (see `archived/ARCHIVED.md`)
- scripts/ — operational and install scripts (platform scripts in `scripts/install/`)
- plugins/ — optional plugins
- config/ — application configuration (not secrets)
- Memory/ — per-user memory profiles
- tests/ — pytest suite
- docs/ — documentation
- gui/ — React + Electron desktop GUI source

### Root-level `.py` files (27 total)

Active entry points and developer utilities (6):

- rex_loop.py — full voice loop entry point (wake word → STT → LLM → TTS)
- rex_speak_api.py — Flask TTS API with auth and rate limiting
- wsgi.py — WSGI entry point for `rex-gui`
- setup.py — legacy setuptools stub (packaging via `pyproject.toml`)
- sitecustomize.py — Windows UTF-8 encoding fix applied at interpreter start
- conftest.py — pytest root conftest (shared fixtures)

Deprecated compatibility shims (4):

- voice_loop.py — legacy voice loop helpers (DeprecationWarning; canonical: `rex/voice_loop.py`)
- config.py — config shim (re-exports from `rex.config`)
- llm_client.py — LLM client shim (re-exports from `rex.llm_client`)
- flask_proxy.py — legacy Flask API and proxy; canonical replacement is `rex-gui`; archived copy at `archived/compat_shims/flask_proxy.py`

Bridge compatibility wrappers (17) — exec canonical `bridge/<name>.py` in their namespace for test-import compatibility; Electron resolves bridges from `bridge/` directly and does not use these root wrappers:

- rex_chat_bridge.py
- rex_chat_stream_bridge.py
- rex_file_extract_bridge.py
- rex_memories_bridge.py
- rex_reminders_bridge.py
- rex_shopping_list_bridge.py
- rex_speaker_bridge.py
- rex_stt_bridge.py
- rex_tasks_bridge.py
- rex_voice_bridge.py
- rex_voice_enrollment_bridge.py
- rex_voice_sample_bridge.py
- rex_voice_upload_bridge.py
- rex_voices_bridge.py
- rex_wakeword_list_bridge.py
- rex_wakeword_sample_bridge.py
- rex_wakeword_train_bridge.py

### Important subpackages

- rex/commands/ — CLI command modules, one per domain (US-REM-027); `rex/cli.py` keeps parser registration, `main()`, and re-exports, and `rex.cli.<name>` remains the import/monkeypatch surface for handlers and service getters
- rex/voice/ — voice pipeline modules, one per concern (US-REM-028); `rex/voice_loop.py` is the facade and `rex.voice_loop.<name>` remains the import/monkeypatch surface (settings, lazy importers, sa/sd, pipeline classes)
- gui/src/main/ — Electron main-process modules, one per concern (US-REM-029); `index.ts` is a thin entrypoint (app lifecycle wiring only), `ipc.ts` aggregates handler registration, IPC handlers live in `gui/src/main/handlers/`, and settings/integration/HA logic lives in `configStore.ts`, `aiSettings.ts`, `voiceSettings.ts`, `settingsDefaults.ts`, `settingsMirror.ts`, `homeAssistant.ts`, `integrationStatus.ts`, `integrationInventory.ts`, `window.ts`
- rex/email_backends/
- rex/calendar_backends/
- rex/messaging_backends/
- rex/dashboard_store.py
- rex/dashboard/sse.py
- rex/identity.py
- rex/voice_identity/
- rex/computers/
- rex/mobile_api/ — authenticated mobile API gateway (issue #323): injectable Flask app factory (`app.py`), typed config helpers, idempotent `users.db` migrations (`db.py`), per-device sessions + rotating hashed refresh tokens (`sessions.py`), short-lived access JWTs + request principal (`auth.py`), structured mobile errors (`errors.py`), truthful runtime-aware capabilities (`capabilities.py`), cross-transport `(user_id, message_id)` chat idempotency (`idempotency.py`, `mobile_message_requests` table), canonical Assistant adapter (`chat.py` — explicit `active_user_id`, never direct `LanguageModel` calls), canonical snake_case streaming events (`events.py`), first-frame-auth WebSocket protocol via Flask-Sock (`websocket.py`, close codes 4401/4403/4408/4429), STT/TTS adapters reusing the existing Whisper and XTTS/edge-tts/pyttsx3 stacks (`voice.py`), routes under `rex/mobile_api/routes/` (`chat.py`: POST /mobile/chat + SSE /mobile/chat/stream; `voice.py`: /mobile/voice/upload + /mobile/tts/playback). Home Assistant/notifications/approvals/tasks/workflows/audit/settings remain explicit 501 scaffolds with false capabilities; `live_voice` stays false. Cross-repo wire contract fixtures live in `tests/mobile_api/contract_vectors.json` (identical copy in the AskRex mobile repo).

### Assistant architecture

`Assistant.generate_reply()` is a thin orchestrator. The pipeline is:

```
Assistant → ContextBuilder → IntentRouter → ActionDispatcher → ResponseBuilder
```

| Component | Module | Responsibility |
|---|---|---|
| `Assistant` | `rex/assistant.py` | Orchestration, lazy-getter init, session state |
| `ContextBuilder` | `rex/context/builder.py` | Assembles system prompt, chat history, user facts |
| `IntentRouter` | `rex/intent/router.py` | Pre-LLM shortcuts: time/date, greetings, capability queries, pending suggestions |
| `ActionDispatcher` | `rex/actions/dispatcher.py` | Skill invocation, HA routing, tool dispatch, LLM call |
| `ResponseBuilder` | `rex/response/builder.py` | Cache lookup/write, TTS cleaning, followup prompts |

Helper functions extracted from `Assistant`:

- `rex.followup_engine.init_followup_engine(settings, user_id)` — initialises the followup engine and returns `(engine, pending_prompt)`.

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

### AppConfig sub-config access pattern

`AppConfig` exposes seven typed sub-config objects. **Always use the nested path** in new code:

| Sub-config | Example field access |
|------------|----------------------|
| `config.audio` | `config.audio.sample_rate`, `config.audio.input_device` |
| `config.voice` | `config.voice.tts_engine`, `config.voice.whisper_device`, `config.voice.wakeword_model` |
| `config.llm` | `config.llm.llm_provider`, `config.llm.model_name`, `config.llm.ollama_url` |
| `config.tools` | `config.tools.tool_timeout`, `config.tools.enabled_tools` |
| `config.integrations` | `config.integrations.home_assistant_base_url`, `config.integrations.openclaw_gateway_url` |
| `config.ui` | `config.ui.gui_port`, `config.ui.gui_host` |
| `config.security` | `config.security.api_key_env`, `config.security.rate_limit_per_minute` |

The mobile gateway adds an eighth typed group, `config.mobile_api`
(`MobileApiConfig`, JSON group `mobile_api` in `config/rex_config.json`):
host/port, token TTLs, body limits, deny-by-default CORS origins,
route-specific rate-limit strings, and `idempotency_retention_hours` (the
retention window for cross-transport chat idempotency records, default 48).
It is canonical nested config with no flat equivalents. The mobile JWT signing secret is `REX_JWT_SECRET` in `.env`
(minimum 32 characters; the gateway fails closed without it). See
`docs/mobile/MOBILE_API_SETUP_WINDOWS.md` for setup.

Flat top-level fields (e.g. `config.llm_provider`, `config.tts_voice`) still work but emit
`DeprecationWarning: Use config.<group>.<field> instead`. Migrate call sites to the nested path
when you touch them. Do not add new flat field reads.

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
- Prefer correct, complete implementations over minimal ones.
- Use appropriate data structures and algorithms — don't brute-force what has a known better solution.
- When fixing a bug, fix the root cause, not the symptom.
- If something I asked for requires error handling or validation to work reliably, include it without asking.

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
- Direct Ruff and Black installations in CI must use the same revisions as `.pre-commit-config.yaml`; never install unpinned formatters in a required check.
- The repository dependency security gate must audit the local project explicitly with `pip-audit --strict .`; a bare `pip-audit` audits the runner environment and is not an acceptable project gate.
- Python releases use `release-please-config.json` plus `.release-please-manifest.json` with the `python` release strategy. Keep the manifest and `pyproject.toml` package version synchronized.
- Session/user state on long-lived components wired into `Assistant` (engines, caches, in-memory logs) must be keyed by `user_id` in a dict, never held as plain instance attributes — one `Assistant` serves multiple identified users, and each request's identity is resolved once (`_resolve_request_user_id`) and passed explicitly as a function argument to every component. Never propagate a request identity by mutating `self._user_id`: shared mutable identity races across overlapping requests. Mirror the `FollowupEngine`/`SuggestionEngine` pattern: every stateful public method takes an explicit `user_id`, validates it via `rex.identity.validate_user_id`, and fails closed (no-op, never a default-user fallback) on missing or invalid identity.
- User IDs are authorization keys, not display strings. Validate them with `rex.identity.validate_user_id` before any path, cache, credential, database, or event access; never sanitize an invalid user ID into a valid one.
- `Assistant` never invents an identity. `Assistant()` is an explicitly unbound instance: it does not assign `"default"`, does not inherit `settings.user_id`, and performs no user-scoped reads or writes at construction (no history preload, no follow-up session, no per-user cache/credential access). Private operations (intent shortcuts, cache lookup, greetings and other early returns, history, context, tool/action dispatch, streaming, completion recording) require an explicit validated identity — the bound constructor `user_id` or a per-request `active_user_id` — and fail closed with `rex.assistant_errors.IdentityRequiredError` otherwise. `user_id="default"` is a valid explicit profile selection only, never an automatic fallback. First-party single-user entrypoints resolve their profile outside `Assistant` via `rex.identity.resolve_entrypoint_user_id(settings, explicit_user=...)` and pass it to `Assistant(user_id=...)`.

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
