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
- INTEGRATIONS_STATUS.md
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
- `master` is not branch-protected. Do not use `gh pr merge --auto`; it can merge immediately while long checks are still running. Issue a merge command only after independently verifying all required GitHub checks are green on the exact PR head.
- See `CONTRIBUTING.md` for full branching model details.

## Tech Stack and Conventions

### Language and runtime

Language: Python 3.11

Packaging: pyproject.toml with setuptools backend  
Developer/operator source install: `pip install .`

Current install/runtime compatibility policy:

- End-user install path: packaged Windows Electron installer with managed Python 3.11 Voice runtime
- Developer/operator install path: Python 3.11 and `pip install .`
- The base install includes `tzdata` so IANA city/timezone tools work on Windows, where `zoneinfo` has no OS timezone database.
- Full Windows GPU + TTS path: Python 3.11 with `requirements-gpu-cu124.txt`
- Do not claim Python 3.12+ support unless the dependency stack has been validated end-to-end and docs, CI, and package metadata are updated together

Entry points:

- rex -> rex.cli:main  # first-class: primary CLI
- rex-gui -> rex.gui_app:main  # developer-only Flask API/dashboard; not used by Electron
- rex-config -> rex.config:cli  # utility: config inspection and migration
- rex-speak-api -> rex_speak_api:main  # backend service: TTS API with auth and rate limiting
- rex-agent -> rex.computers.agent_server:main  # backend service: optional remote PC control API
- rex-tool-server -> rex.openclaw.tool_server:main  # backend service: OpenClaw tool adapter at /rex/tools/{tool_name}

Mobile API gateway (issue #323) runs via the CLI, not a console script:

- python -m rex mobile-api [--host HOST] [--port PORT]  # authenticated mobile gateway (default 127.0.0.1:8765)
- python -m rex mobile-user create --username NAME      # safe mobile user creation (getpass prompts)

### Core components

API: Flask (Flask-CORS, Flask-Limiter, Flask-Sock for the mobile WebSocket)

GUI: React + Electron under `gui/` is the primary packaged interface. `rex.gui_app` is a developer-only Flask API/dashboard and is not spawned by Electron. Archived Tkinter files are unsupported.

- `gui/src/pages/SettingsPage.tsx` is a thin Settings category/router facade. Section implementations, controllers, and helpers belong under `gui/src/pages/settings/`; keep each settings module below 1,000 lines and preserve the structural regression in `gui/tests/settingsSections.test.ts`.

Config: Pydantic v2, python-dotenv

STT: OpenAI Whisper (offline)

Wake word: openWakeWord

- openWakeWord models are stream-oriented. Keep Rex capture windows independent from model inference chunks: marked openWakeWord models must be evaluated in 1,280-sample / 80 ms chunks at 16 kHz and aggregate the peak score across the longer capture window. Do not pass an arbitrary one-second frame as one model prediction.
- US-046 wake-word reliability is a deliberate exception to the general no-binary-test-artifacts rule: only `tests/fixtures/wakeword/` may contain the small tracked synthetic WAV corpus. `openwakeword==0.6.0` belongs in the dev extra so the standard CI suite evaluates that corpus with the same detector version used for the tracked report; do not move the full `ml` extra into base CI.
- Lazy optional-dependency caches must follow live module state, not stale test aliases. For openWakeWord, prefer the explicit compatibility alias while it is present, otherwise use `sys.modules`, and refresh the import when neither is live; do not return a cached fake module after monkeypatch teardown.

TTS: Coqui XTTS (voice cloning supported)

Optional TTS:
- edge-tts
- pyttsx3

LLM providers:

- local Transformers
- OpenAI API
- OpenRouter (OpenAI-compatible API with a separate credential and model slug)
- Ollama

Search providers:

- SerpAPI
- Brave
- Google CSE
- DuckDuckGo

Home Assistant safety:

- `lock`, `cover`, `alarm_control_panel`, broad `script.*`, and broad `scene.*` mutations are sensitive and must use the canonical short-lived confirmation-token gate before dispatch. The first unconfirmed call must have no side effect. Keep `automation`, `python_script`, `shell_command`, `update`, and unknown domains fail-closed unless a later reviewed policy explicitly changes them.

### Style and quality

- `PRD-production-readiness.md` is the single authoritative tracker for the current integrated production-readiness/Rex 2.0 work. For remaining work, follow its dated `Integrated execution order` rather than raw story file position. Planned Rex 2.0 contracts are not implemented behavior until their individual story is merged and verified.

- `rex.runtime` is the canonical interface-agnostic turn contract layer. `TurnContext` owns immutable validated identity/scope, origin/response mode, monotonic timing/deadline, and authorization snapshot references; `TurnEventStream` owns ordered correlated events and exactly one terminal outcome; `TurnEngine` preserves wrapped return/exception behavior while emitting those events. `Assistant.generate_reply()` now runs through this engine; `stream_reply()` remains on the legacy streaming path until US-096, and interface adapter parity remains unclaimed until US-097.

- Prefer clear, testable functions over clever code.
- Keep changes small and reviewable.
- Add logging for non-trivial behavior.
- Update or add tests when behavior changes.
- Avoid introducing new heavy dependencies unless clearly justified.

### Security

Never commit secrets.

Desktop secrets (API keys, tokens, passwords) belong in the OS-backed credential
vault (`rex.credential_vault`, Windows DPAPI), not in plaintext `.env` or JSON.
Plaintext environment/config reads are disabled by default. Unpackaged
operator and CI runs may explicitly opt in with
`REX_ALLOW_PLAINTEXT_CREDENTIAL_FALLBACK=1`; packaged Electron strips and
rejects that flag. New writes always go through the vault. See "Credential
vault (S4)" below.

Runtime settings belong in:

config/rex_config.json

Core config, `.env`, profiles, and persistent data paths must resolve
through `rex.runtime_paths`, not the process working directory. Electron must
launch every Python bridge with `bridgeSpawnOptions()` so development uses the
repository root and packaged builds use Electron `userData`.

Persistent data is partitioned explicitly:

- `data/household/` for shared service state, authentication, notifications,
  schedules, and conversation-history databases whose rows are user-scoped.
- `data/users/<validated-user-id>/` for private memory and learned autonomy
  preferences/history.

`REX_DATA_DIR` remains a compatibility override for the exact shared-data root.
Managed Electron launches also set `ASKREX_HOUSEHOLD_DATA_DIR` and
`ASKREX_USERS_DATA_DIR`. New persistence code must use `rex.runtime_paths`.
Existing `data/` and `~/.rex` stores migrate with
`scripts/migrate_runtime_data.py`, which is dry-run by default and must never
overwrite conflicts. Never write runtime state beneath `bridge/`, the
application archive, or packaged resources.

#### Desktop user profile authority

- `rex.user_profile_service` is the canonical composition layer for desktop
  profile identity, private preferences, live permissions/role, voice
  enrollment metadata, and avatar metadata.
- Electron profile operations use `bridge/rex_profile_bridge.py` through the
  typed main-process handlers. The renderer never supplies or selects a user
  ID; the immutable authenticated Electron session is authoritative.
- Profile avatars are private user data under
  `data/users/<validated-user-id>/profile/avatar.jpg`. Inputs are bounded,
  validated as JPEG/PNG, normalized, and never exposed as filesystem paths.
- Shared household settings stay outside profile storage and remain managed
  through Settings. Do not duplicate household configuration into profile JSON.
- SMS remains a direct route/backend for compatibility but is intentionally
  absent from primary navigation. Settings has one persistent bottom shortcut,
  not a duplicate scrolling entry.

#### Credential vault (S4)

`rex/credential_vault.py` is the Windows DPAPI-backed credential vault.
Values are encrypted at rest (`win32crypt.CryptProtectData`/`CryptUnprotectData`
via `pywin32`); config files only ever hold the vault *key*, never a secret
value.

- Two scopes, mirroring the household/private data split above:
  `scope="household"` (installation-wide secrets — OpenAI/HA/search keys;
  default, matches pre-vault global behavior) and `scope="user"` + a
  validated `user_id` (bound to one Rex profile, e.g. a personal email
  account `credential_ref`). Each scope encrypts with different DPAPI
  entropy, so one Rex user cannot decrypt another's entries even when both
  share one Windows login.
- Storage: `<household_data_dir()>/credentials/vault.json` or
  `<user_data_dir(user_id)>/credentials/vault.json` (via `rex.runtime_paths`).
- `CredentialManager` (`rex/credentials.py`) is vault-only by default. In
  explicit unpackaged legacy/operator mode, process environment and legacy
  config may override the vault. `set_token(..., persist=True)` always writes
  through to the vault and raises `VaultUnavailableError` if unavailable.
- `rex/config.py`'s direct `os.getenv(...)` secret reads for `AppConfig`
  (`HA_TOKEN`, `OPENAI_API_KEY`, etc.) go through `_secret_env_or_vault()`,
  which applies the same vault-only default and explicit legacy-mode policy.
- Electron never performs vault cryptography. `bridge/rex_credential_vault_bridge.py`
  is the only path Electron uses to reach it (stdin/stdout JSON, like every
  other bridge); `gui/src/main/credentialVault.ts` wraps that bridge call.
- Non-Windows dev/CI: `get_credential_vault()` raises `VaultUnavailableError`
  (no implicit fallback). `InMemoryCredentialVault` exists only for tests —
  never selected by a production code path.
- One-time migration of existing plaintext `.env` / `config/credentials.json`
  secrets: `scripts/migrate_credentials_to_vault.py` (dry-run by default; <!-- pragma: allowlist secret -->
  `--apply` verifies the vault write and opaque-reference registry before
  atomically sanitizing the source). It never creates plaintext backups or
  secret-derived output; rollback references remain encrypted in the vault
  with a secret-free recovery journal.

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
- bridge/ — canonical Electron stdin/stdout JSON bridge processes (plus integration adapters)
- archived/ — retired files kept for reference; not maintained (see `archived/ARCHIVED.md`)
  - `archived/flask_dashboard/tests/` preserves retired Flask-dashboard placeholder tests and is outside pytest's active `tests/` collection.
- scripts/ — operational and install scripts (platform scripts in `scripts/install/`)
- plugins/ — optional plugins
- config/ — application configuration (not secrets)
- Memory/ — per-user memory profiles
- tests/ — pytest suite
- docs/ — documentation
- gui/ — React + Electron desktop GUI source

### Root-level `.py` files (27 total)

Active entry points and developer utilities (6):

- rex_loop.py — source voice-loop entry point; defaults to Hold-to-Talk/manual activation, with beta wake-word only via `--mode wake-word`
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
- rex/tools/execution.py — canonical typed tool lifecycle; all registered dispatch must pass availability, argument, identity, permission, risk, confirmation, execution, normalization, independent verification, truthful response, and redacted audit stages. Read-only success is `completed`; mutation success is `verified` only.
- `rex/latency.py` / `rex/rexbench.py` - privacy-safe request timing and p50/p95 aggregation. Timing telemetry must never include prompts, transcripts, user IDs, memory contents, credentials, or tool payloads; see `docs/performance.md`.
- gui/src/main/ - Electron main-process modules, one per concern (US-REM-029); `index.ts` is a thin entrypoint (app lifecycle wiring only), `ipc.ts` aggregates handler registration, IPC handlers live in `gui/src/main/handlers/`, integration credential persistence/rollback lives in `integrationSettingsStorage.ts`, and settings/integration/HA logic lives in `configStore.ts`, `aiSettings.ts`, `voiceSettings.ts`, `settingsDefaults.ts`, `settingsMirror.ts`, `homeAssistant.ts`, `integrationStatus.ts`, `integrationInventory.ts`, `window.ts`
- `gui/src/pages/settings/integrations/` owns the Settings > Integrations controller and focused UI components; keep OpenClaw token handling renderer-blind and route all secret persistence through the main-process vault helpers.
- `gui/src/types/settingsRouting.ts` is the shared parser for Settings deep links such as `#/settings?section=integrations`; invalid or missing sections fail safely to General.
- Installed Electron artifact smoke may override Electron `userData` only when `ASKREX_ARTIFACT_SMOKE=1` and `ASKREX_ARTIFACT_SMOKE_RUNTIME_ROOT` is set; the PowerShell harness must point Python `ASKREX_RUNTIME_DIR` and Electron userData at the same isolated temp runtime root.
- rex/credential_vault.py — Windows DPAPI-backed credential vault (S4); see "Credential vault (S4)" above
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

`Assistant.generate_reply()` is the canonical non-streaming TurnEngine entry point. Identity is validated before the turn begins; all subsequent intent/cache/model routing, context building, action/tool dispatch, result verification, response building, and history recording run inside one correlated turn. `stream_reply()` is not yet migrated (US-096). The non-streaming pipeline is:

```
Assistant → TurnEngine → IntentRouter/Cache → ContextBuilder → ActionDispatcher → ResponseBuilder → History
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

Health/readiness check:

python -m rex doctor

Lightweight process liveness check (used by the developer-only Docker image):

python -m rex doctor --healthcheck

Docker is developer/operator-only and is not an end-user or supported production deployment path.

Release health gate:

python -m rex doctor --release-gate
python scripts/security_audit.py --release-gate

Deterministic latency baseline (instrumentation/framework evidence only):

```powershell
python scripts/rexbench.py --profile baseline --iterations 20 --output docs/performance/rexbench-baseline.json
```

`deterministic_mock` RexBench results never prove live provider, model, network, audio, or hardware latency is within budget. Use `docs/performance.md` and the final live RexBench release gate for production claims.

Electron identity (required before launch):

rex identify --user <id>

Legacy Electron task/history ownership migration is dry-run first:

python scripts/migrate_electron_data_ownership.py --user <id>
python scripts/migrate_electron_data_ownership.py --user <id> --apply

Credential vault migration (moves plaintext `.env`/`config/credentials.json`
secrets into the OS-backed vault) is dry-run first:

python scripts/migrate_credentials_to_vault.py --scope household --owner household
python scripts/migrate_credentials_to_vault.py --scope household --owner household --apply

Text mode:

python -m rex

Voice mode:

python rex_loop.py --mode hold-to-talk
# beta opt-in only:
python rex_loop.py --mode wake-word

The source CLI defaults to Hold-to-Talk/manual activation and must not initialize the wake-word detector unless `--mode wake-word` is explicitly selected. Electron Hold-to-Talk is the supported production voice path. Canonical voice-stage timing events are documented in `docs/voice_pipeline.md`; preserve those event names and their `session_id`/monotonic timing fields when changing the pipeline. It runs
renderer recording -> persistent managed Whisper STT -> streamed assistant
response -> configured TTS -> selected output-device playback. Preserve
cancellation/barge-in, replay, microphone device-loss fallback, repeated turns,
stage-specific errors, and structured timing events. The Voice-page microphone
selector must route to both Hold-to-Talk and the Python wake-word capture path;
never assume a Chromium device selection automatically changes PortAudio's
default device. Wake-word mode remains beta unless it is verified on physical
audio hardware.

GUI:

rex-gui

TTS API:

python rex_speak_api.py

## Test and Lint

Run tests:

pytest -q

Targeted tests:

pytest -q tests/<file>.py

Skipped-test budget gate (the report must come from the primary CI marker set):

```powershell
pytest -m "not slow and not audio and not gpu" -rs -q | Tee-Object pytest.out
python scripts/check_skip_budget.py pytest.out
```

The runtime budget is `scripts.check_skip_budget.SKIP_BUDGET`. When skips are
removed, lower the budget in the same PR. Never raise it without updating
`docs/testing/SKIPPED-TESTS-INVENTORY.md` with evidence and rationale.

Validate the source-site inventory after adding, removing, or moving any pytest
skip call/decorator:

```powershell
python scripts/check_skip_inventory.py
```

Every inventory row must use one action: `keep`, `fix`, `replace`, or `archive`.
Permanent guards need a written rationale; non-trivial actions need a non-circular
`US-###` follow-up. Update the inventory in the same PR as skip-site changes.

Electron GUI quality gates (run from `gui/`):

```powershell
npm.cmd ci
npm.cmd run lint
npm.cmd run typecheck
npm.cmd test -- --run
npm.cmd run build
npm.cmd audit --audit-level=high
```

The GUI uses the flat ESLint configuration in `gui/eslint.config.mjs`.

Windows Electron distribution (run from `gui/`):

```powershell
npm.cmd run runtime:build   # managed Python 3.11 Voice profile
npm.cmd run dist            # runtime + GUI + NSIS installer
```

Packaged Electron always resolves `resources/python/python.exe`; it must never
fall back to machine Python or a checkout `.venv`. The installer bundles the
AskRex wheel, canonical `bridge/` scripts, pinned Voice dependencies, and
bundled FFmpeg. Validate packages with
`scripts/verify_electron_package_contents.py` and
`scripts/test_installed_electron_artifact.ps1`. Flask and user configuration,
credentials, profiles, memories, transcripts, and logs are forbidden package
contents. Generated runtimes live under ignored `gui/runtime/`.

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

Secrets → OS-backed credential vault (plaintext only in explicit unpackaged legacy mode)
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
It is canonical nested config with no flat equivalents. The mobile JWT signing secret is vault entry `REX_JWT_SECRET`
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

Integration readiness uses the shared state vocabulary in `rex/integration_state.py` and
`gui/src/types/ipc.ts`: unavailable, unconfigured, configured, reachable, authenticated,
degraded, read-only, write-capable, write-tested, and verified. Credentials alone mean
`configured`, never connected or authenticated. Use `rex integrations` for the CLI inventory;
`rex doctor` includes the same evidence without making live-provider claims.

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
- Never dispatch a mutating canonical tool by calling its handler directly. Use `ToolDispatcher.dispatch()` / `ToolExecutionLifecycle`; HTTP acceptance or a returned object is `attempted_unverified` until an independent verifier succeeds.
- The canonical wake-word implementation is `rex/wakeword/` (`rex.wakeword.utils`, `rex.wakeword.listener`). Root-level `wakeword_utils.py` and `wakeword_listener.py` were stale re-exports and have been deleted. Use `rex.wakeword_utils` (package shim) or `rex.wakeword.utils` directly.
- Direct Ruff and Black installations in CI must use the same revisions as `.pre-commit-config.yaml`; never install unpinned formatters in a required check.
- The repository dependency security gate must audit the local project explicitly with `pip-audit --strict .`; a bare `pip-audit` audits the runner environment and is not an acceptable project gate.
- Python releases use `release-please-config.json` plus `.release-please-manifest.json` with the `python` release strategy. Keep the manifest and `pyproject.toml` package version synchronized.
- Session/user state on long-lived components wired into `Assistant` (engines, caches, in-memory logs) must be keyed by `user_id` in a dict, never held as plain instance attributes — one `Assistant` serves multiple identified users, and each request's identity is resolved once (`_resolve_request_user_id`) and passed explicitly as a function argument to every component. Never propagate a request identity by mutating `self._user_id`: shared mutable identity races across overlapping requests. Mirror the `FollowupEngine`/`SuggestionEngine` pattern: every stateful public method takes an explicit `user_id`, validates it via `rex.identity.validate_user_id`, and fails closed (no-op, never a default-user fallback) on missing or invalid identity.
- User IDs are authorization keys, not display strings. Validate them with `rex.identity.validate_user_id` before any path, cache, credential, database, or event access; never sanitize an invalid user ID into a valid one.
- `Assistant` never invents an identity. `Assistant()` is an explicitly unbound instance: it does not assign `"default"`, does not inherit `settings.user_id`, and performs no user-scoped reads or writes at construction (no history preload, no follow-up session, no per-user cache/credential access). Private operations (intent shortcuts, cache lookup, greetings and other early returns, history, context, tool/action dispatch, streaming, completion recording) require an explicit validated identity — the bound constructor `user_id` or a per-request `active_user_id` — and fail closed with `rex.assistant_errors.IdentityRequiredError` otherwise. `user_id="default"` is a valid explicit profile selection only, never an automatic fallback. First-party single-user entrypoints resolve their profile outside `Assistant` via `rex.identity.resolve_entrypoint_user_id(settings, explicit_user=...)` and pass it to `Assistant(user_id=...)`.
- Vault (`rex.credential_vault`) failures fail closed. Read paths return an absent credential only when the vault is unavailable; corrupt schema, metadata, scope, account, slot, owner, or reference data raises. Plaintext config/environment is consulted only when explicit legacy mode is enabled outside packaged Electron. Write paths propagate vault, readback, registry, and mirror failures so the GUI cannot report false success.
- An Electron `ipcMain.handle` callback returning `{ ok: true, someList: buildX() }` where `buildX()` is `async` is a silent bug, not a type error: TypeScript happily infers the handler's return type around a nested `Promise`, but the renderer receives an unresolved promise instead of the array. `tsc --noEmit` will not catch this — grep every call site of a function you just made `async` and confirm each one added `await`, don't rely on the type checker alone.

## OpenClaw Migration Status

Rex integrates with OpenClaw over HTTP (not as a Python package). Key facts:

- both OpenClaw flags default to `False`; OpenClaw is experimental/off by default. Enabling either gateway-backed path requires a valid HTTP(S) gateway URL plus `OPENCLAW_GATEWAY_TOKEN`, and incomplete enabled configuration fails closed.
- Gateway health is explicit: `/healthz` may establish reachability only. It does not prove authentication or tool capability.

- Phase 8 (HTTP integration) is complete. All `find_spec("openclaw")` / `import openclaw` stubs have been removed and replaced with HTTP client calls.
- OpenClaw adapters live in `rex/openclaw/`: `agent.py`, `tool_bridge.py`, `event_bridge.py`, `browser_bridge.py`, `voice_bridge.py`, `http_client.py`, `tool_server.py`, and tool handlers under `rex/openclaw/tools/`.
- HTTP client: `rex/openclaw/http_client.py` (`OpenClawClient`) handles auth, retries, timeouts for all gateway calls. Singleton via `get_openclaw_client(config)`.
- Config fields: `openclaw_gateway_url`, `openclaw_gateway_timeout`, `openclaw_gateway_max_retries` in `AppConfig`; `OPENCLAW_GATEWAY_TOKEN` in the credential vault.
- Feature flag `use_openclaw_voice_backend` in `AppConfig` (config path: `openclaw.use_voice_backend`): when True, voice loops swap `Assistant` for `VoiceBridge`, routing LLM calls through OpenClaw's `/v1/chat/completions`.
- Feature flag `use_openclaw_tools` in `AppConfig` (config path: `openclaw.use_tools`): when True, `ToolBridge.execute_tool()` dispatches to OpenClaw's `/tools/invoke`; 404 uses the local tool, while connection/auth/429/5xx failures fall back locally with a structured warning after bounded retries. A 403 remains a hard policy denial.
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

## Authoritative Product Planning

For product scope, delivery order, and non-negotiable behavior, read these before planning or implementation work:

- `docs/planning/source-of-truth/REX_Unified_Build_Spec_UPDATED.md`
- `docs/planning/source-of-truth/REX_ACTIVE_CHECKLIST.md`
- `docs/planning/TEAM_LEAD_OPERATING_RULES.md`

The first two files are the product sources of truth. Other PRDs are supporting history and feature inputs only. Do not mark a feature complete from a checklist alone; verify current code, tests, packaged behavior, and user-visible truth.

#### Mobile device pairing authority (S5)

- `rex/mobile_api/pairing.py`, `device_proof.py`, and `grants.py` implement the desktop-owned P-256 pairing authority.
- Password login never creates a device grant. The mobile HTTP API exposes only proof submission and private-token status polling; challenge creation, approval, denial, listing, and revocation remain local Electron IPC operations through `bridge/rex_pairing_bridge.py`.
- Challenges expire after 120 seconds and are single-use. The v2 proof transcript binds desktop ID, challenge/nonce, canonical public key, user, scopes, one-time code, advertised HTTPS URL, certificate fingerprint, and SPKI pins.
- Persisted grants are immutable/versioned, expiring, revocable, and audited.
- S6 session enforcement lives in `rex/mobile_api/authorization.py`, `auth.py`, and `sessions.py`. Password login is bootstrap-only with zero scopes. Device activation requires a short-lived P-256 proof challenge and atomically replaces/revokes the bootstrap session family.
- Never authorize from JWT/client-supplied scopes or device metadata. Every request/refresh must resolve the current device, latest grant version, desktop/user binding, scopes, expiry, and revocation from SQLite. Long-lived SSE/WS output must revalidate while streaming.
- Mobile authorization is the intersection of the immutable device grant and live Rex user permissions. Home scopes require `ha_control` or `admin`; approval responses require `admin`. Permission metadata or capability tags must never widen authority.
- Device revocation and grant supersession revoke all bound sessions and refresh families. Existing device keys cannot be reassigned across Rex users.
- Pairing key possession is not strong authentication. Never stamp `strong_auth_at` during pairing/session activation.
- S8 (`rex/mobile_api/strong_auth.py`, `routes/strong_auth.py`, `routes/home.py`): high/critical mobile actions require a short-lived P-256 device assertion bound to the exact canonical action hash, authenticated session/user, paired device, immutable grant/version, desktop ID, server-owned risk level, scope, nonce, and expiry. Verification issues a second short-lived approval ID that is consumed atomically once at the actual execution boundary. A recent biometric timestamp, client risk label, or reused approval never authorizes another action. Home Assistant mobile commands return `verified`, `attempted_unverified`, `denied`, or `failed` truthfully; approval consumption does not imply the action succeeded. Physical Face ID/passcode and iPhone integration remain mobile-repo/hardware gates.
- S7 (`rex/mobile_api/tls.py`): any non-loopback mobile API bind always requires usable TLS; `mobile_api.require_tls` only opts a *loopback* bind into TLS for local testing and cannot weaken the non-loopback boundary. `MobileApiServices.build()` provisions or reuses one long-lived self-signed P-256 certificate under `<household_data_dir>/mobile_tls/` and fails closed with `MobileTlsConfigurationError` when material cannot be generated or loaded. The advertised HTTPS URL, SHA-256 certificate fingerprint, and SPKI pins are included in every pairing QR payload, signed by pairing proof transcript v2, returned by approved `/mobile/pairing/status`, and persisted immutably on device/grant records. `create_mobile_app()` independently refuses a TLS-required injected service container without material, and a TLS-owned app rejects plaintext requests with `TLS_REQUIRED`. `POST /mobile/auth/activate-device` fails closed (`PAIRING_INVALID`) when the current gateway-owned transport binding differs from the approved device/grant binding (rotated/reset certificate, changed endpoint, or a pre-S7 unbound legacy device). Loopback-only development remains unaffected unless explicitly opted into TLS. Actual client-side pin validation and physical LAN/phone hardware validation live in the separate mobile repo and are not implemented or exercised here.
- See `docs/mobile/DEVICE_PAIRING.md`, `docs/mobile/STRONG_AUTH.md`, `docs/mobile/MOBILE_API_SETUP_WINDOWS.md`, `tests/mobile_api/test_pairing.py`, `tests/mobile_api/test_grant_enforcement.py`, `tests/mobile_api/test_strong_auth.py`, `tests/mobile_api/test_home_strong_auth.py`, `tests/mobile_api/test_tls.py`, and `tests/mobile_api/test_transport_binding.py`.
