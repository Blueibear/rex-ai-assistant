# CLAUDE.md

## Project Overview
Rex AI Assistant is a local-first, voice-activated AI companion. It supports wake word detection, speech-to-text, LLM chat, and text-to-speech, with optional integrations for search, messaging, email, calendar, and Home Assistant.

Primary goals: reliability, security by default, and smooth day-to-day usage on Windows 10/11, macOS, and Linux.

## Tech Stack and Conventions

### Language and runtime
- Language: Python 3.9 to 3.13 (3.10+ preferred)
- Packaging: pyproject.toml with setuptools backend, install via `pip install .`
- Entry points:
  - `rex` -> `rex.cli:main`
  - `rex-config` -> `rex.config:cli`
  - `rex-speak-api` -> `rex_speak_api:main`

### Core components
- API: Flask (Flask-CORS, Flask-Limiter)
- Config: Pydantic v2, python-dotenv
- STT: OpenAI Whisper (offline)
- Wake word: openWakeWord
- TTS: Coqui XTTS (voice cloning supported); optional edge-tts or pyttsx3
- LLM: local Transformers, OpenAI API, or Ollama
- Search: plugin-based (SerpAPI, Brave, Google CSE, DuckDuckGo)

### Style and quality
- Prefer clear, testable functions over clever code.
- Keep changes small and reviewable.
- Add logging for non-trivial behavior (use the existing logging utilities).
- Update or add tests when behavior changes.
- Avoid introducing new heavy dependencies unless clearly justified.

### Security
- Never commit secrets. Secrets belong in `.env` only.
- Runtime settings belong in `config/rex_config.json`. Secrets belong in `.env`.
- Prefer least-privilege defaults for any networked feature (TTS API, plugins, integrations).
- Treat all external inputs as untrusted (web content, email, SMS, plugin results).

## Repository Structure
Top-level directories and what they contain:
- `rex/` - main package (CLI, services, workflows, integrations)
- `scripts/` - operational scripts (health checks, utilities, tooling)
- `plugins/` - optional plugins (web search and other extensible features)
- `config/` - app configuration files (not secrets)
- `Memory/` - per-user memory profiles and stored context
- `tests/` - pytest suite
- `docs/` - additional documentation

Notable top-level modules and entrypoints:
- `rex_loop.py` - full voice loop (wake word -> STT -> LLM -> TTS)
- `voice_loop.py` - core voice loop helpers
- `wakeword_listener.py` - wake word listener utilities
- `rex_speak_api.py` - Flask TTS API with auth and rate limiting
- `run_gui.py` / `gui.py` - desktop GUI

Notable subpackages:
- `rex/email_backends/` - email backend adapters (base interface, stub, IMAP/SMTP, multi-account config/router)
- `rex/calendar_backends/` - calendar backend adapters (base interface, stub, ICS read-only)
- `rex/messaging_backends/` - SMS backend adapters (base interface, stub, Twilio, multi-account config, factory)
- `rex/dashboard_store.py` - SQLite-backed dashboard notification store (write/read/query/retention)
- `rex/identity.py` - session-scoped user identity resolution (fallback when voice recognition is unavailable)
- `rex/voice_identity/` - voice speaker recognition scaffolding (types, embeddings store, recognizer, fallback flow, optional dep guards)
- `rex/computers/` - Windows computer control client foundation (config, HTTP client, service; Cycle 5.1)

## Commands

### Install
- Create venv:
  - Windows PowerShell: `python -m venv .venv` then `.\.venv\Scripts\Activate.ps1`
  - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
- Base install: `python -m pip install --upgrade pip setuptools wheel` then `pip install .`
- Optional stacks:
  - CPU ML: `pip install -r requirements-cpu.txt`
  - GPU CUDA 12.4: `pip install -r requirements-gpu-cu124.txt`
  - GPU (alt): `pip install -r requirements-gpu.txt`
  - Dev tools: `pip install -r requirements-dev.txt`

### Run
- Health check: `python scripts/doctor.py`
- Text mode: `python -m rex`
- Voice mode: `python rex_loop.py`
- GUI: `python run_gui.py`
- TTS API: `python rex_speak_api.py` (or `rex-speak-api` if installed as a script)

### Test and lint
- Tests: `pytest -q`
- Targeted tests: `pytest -q tests/<file>.py`

## Setup & Installation (GPU)
- Do not reintroduce GPU extras like `.[gpu-cu118]`, `.[gpu-cu121]`, or `.[gpu-cu124]` unless they are fully functional with the required PyTorch index behavior.
- Supported GPU installs are requirements-file based because CUDA wheels require `--extra-index-url`.
- Keep GPU guidance aligned across `INSTALL.md`, `README.md`, and requirements files in the same PR.

## Integration backend strategy (email/calendar/notifications/sms/voice)
- Email backend target is provider-agnostic IMAP4 over SSL (read) + SMTP STARTTLS/SSL (send).
- Multi-account email support is required for production readiness: account-aware routing, active/default selection, and explicit fallback order.
- Notification priority, digest, quiet-hours, and escalation behavior must remain stable while channel delivery backends are upgraded.
- Calendar backend rollout order: ICS read-only first (safest, easiest to test), then CalDAV/Google OAuth later.
- SMS backend must keep Twilio as optional functionality and must retain safe no-credential failure behavior.
- Voice speaker recognition must be scaffolded behind optional dependencies; default installs must stay lightweight.

## Dependency and lockfile rules (non-negotiable)
- Keep `Pipfile` and `Pipfile.lock` Dependabot and pipenv lock friendly on clean Linux (no CUDA runtime assumptions).
- Do not add heavy ML or CUDA dependencies to `Pipfile` default packages.
- Heavy voice-recognition dependencies must be optional extras only (for example in `pyproject.toml` optional dependency groups) and guarded at runtime.
- Any dependency change for integrations must include explicit lockability verification (`pipenv lock --clear`) in the PR verification notes.

## Email integration config keys (implemented)
- Runtime config section for multi-account email in `config/rex_config.json`:
  - `email.default_account_id`
  - `email.accounts[]` with per-account IMAP/SMTP server settings and `credential_ref`
- Notification routing uses metadata key `email_account_id` for explicit account selection.
- Credentials are split by concern:
  - non-secret server/runtime values in `config/rex_config.json`
  - secrets in `.env` or `config/credentials.json` via `CredentialManager`
- Email backend code lives in `rex/email_backends/` (base, stub, imap_smtp, account_config, account_router).

## Email CLI commands (implemented)
- `rex email accounts list`
- `rex email accounts set-active --account-id <id>`
- `rex email test-connection [--account-id <id>]`
- `rex email send --account-id <id> --to <recipient> --subject <subject> --body <body>`
- `rex email unread [--limit N] [-v]` (pre-existing)
- All email commands accept `--user <id>` to override the active user context.

## Calendar CLI commands (implemented)
- `rex calendar upcoming [--days N] [--conflicts] [-v] [--user <id>]`
- `rex calendar test-connection`

## Calendar backend config keys (implemented)
- `calendar.backend`: `"stub"` (default) or `"ics"`
- `calendar.ics.source`: local `.ics` file path or HTTPS URL
- `calendar.ics.url_timeout`: fetch timeout in seconds (default 15)
- Calendar backend code lives in `rex/calendar_backends/` (base, stub, ics_backend, ics_parser, factory).

## Identity CLI commands (implemented)
- `rex whoami` — show the active user for the session (from session state, config, or explicit flag)
- `rex identify --user <id>` — set the active user non-interactively
- `rex identify` — interactive selection from known Memory/ profiles
- Identity resolution priority: `--user` flag > session state > `runtime.active_user` > `runtime.user_id`
- Session state stored in OS-appropriate temp dir (`rex-ai/session.json`), not persisted to repo.
- Identity module: `rex/identity.py`

## Identity config keys
- `identity.known_users`: reserved for future use (manual user list override)
- `identity.require_user`: `false` (default) — when `true`, commands that need user context will fail if no user is resolved

## Voice identity module (scaffolding)
- Module: `rex/voice_identity/` (types, embeddings_store, recognizer, fallback_flow, optional_deps)
- Pure-Python cosine similarity; no numpy/torch in default install.
- Per-user embeddings stored as JSON at `Memory/<user>/voice_embeddings.json`.
- Recognizer returns `recognized`, `review`, or `unknown` based on configurable thresholds.
- Fallback flow integrates with `rex/identity.py` (session-scoped identity resolution).
- Voice loop (`rex/voice_loop.py`) accepts an optional `identify_speaker` callback.
- Heavy deps (speechbrain, resemblyzer) are in `pyproject.toml` optional extras group `voice-id`.
- Install: `pip install '.[voice-id]'`
- Tests: `pytest -q tests/test_voice_identity_fallback.py tests/test_optional_voice_id_imports.py`
- Docs: `docs/voice_identity.md`

## Voice identity config keys (scaffolding)
- `voice_identity.enabled`: `false` (default) — enable voice identity in the voice loop
- `voice_identity.accept_threshold`: `0.85` — minimum cosine similarity for `recognized`
- `voice_identity.review_threshold`: `0.65` — minimum cosine similarity for `review`
- `voice_identity.embedding_dim`: `192` — expected dimensionality of embedding vectors
- `voice_identity.model_id`: `"synthetic"` — identifier for the active embedding model

## Optional extras policy
- Heavy ML/audio dependencies live in `pyproject.toml` optional extras groups, never in default `[project.dependencies]` or Pipfile `[packages]`.
- Available extras: `audio`, `ml`, `voice-id`, `sms`, `dev`, `devtools`, `test`, `full`.
- Runtime code must guard imports of optional packages and fall back cleanly when missing.
- Pipfile/Pipfile.lock must remain lockable on clean Linux (`pipenv lock --clear`).

## Messaging CLI commands (implemented)
- `rex msg send --channel sms --to <number> --body <text> [--account-id <id>] [--user <id>]`
- `rex msg receive --channel sms [--limit N] [--user <id>]`
- All messaging commands accept `--user <id>` and `--account-id <id>` for routing.

## Messaging backend config keys (implemented)
- `messaging.backend`: `"stub"` (default) or `"twilio"`
- `messaging.default_account_id`: account used when none is specified
- `messaging.accounts[]` with per-account `id`, `label`, `from_number`, `credential_ref`, and `owner_user_id`
- `messaging.accounts[].owner_user_id`: optional user profile ID — inbound SMS to this account's `from_number` is tagged with this user_id at ingest time
- `messaging.inbound.enabled`: `false` (default) — enable the inbound SMS webhook endpoint and store
- `messaging.inbound.auth_token_ref`: `"twilio:inbound"` — credential ref for Twilio auth token used for webhook signature verification
- `messaging.inbound.store_path`: SQLite path for inbound messages (default: `data/inbound_sms.db`)
- `messaging.inbound.retention_days`: days to retain inbound messages (default: 90)
- `messaging.inbound.rate_limit`: rate limit for the webhook endpoint (default: `"120 per minute"`, Flask-Limiter format)
- Inbound webhook endpoint: `POST /webhooks/twilio/sms` (hosted by `flask_proxy.py`)
- Webhook wiring: `register_inbound_sms_webhook()` in `rex/messaging_backends/webhook_wiring.py` reads config, resolves credentials, and registers the blueprint at startup
- Reverse proxy: when behind a reverse proxy, configure `ProxyFix` and `REX_TRUSTED_PROXIES` so `request.url` matches the externally visible URL for Twilio signature verification
- Credentials are split by concern:
  - non-secret config (from_number, routing) in `config/rex_config.json`
  - secrets in `.env` or `config/credentials.json` via `CredentialManager`
- Inbound store migration: existing SQLite databases are auto-migrated to add the `user_id` column if absent (idempotent `ALTER TABLE` via `PRAGMA table_info` check at startup)
- Messaging backend code lives in `rex/messaging_backends/` (base, stub, twilio_backend, account_config, factory, inbound_store, inbound_webhook, twilio_signature, webhook_wiring).

## Notification dashboard config keys (implemented)
- `notifications.dashboard.store.type`: `"sqlite"` (only supported type)
- `notifications.dashboard.store.path`: database file path (default: `data/dashboard_notifications.db`)
- `notifications.dashboard.store.retention_days`: days to retain (default: 30)
- Dashboard API endpoints: `GET /api/notifications`, `POST /api/notifications/<id>/read`, `POST /api/notifications/read-all`
- Dashboard store code: `rex/dashboard_store.py`

## Windows computer control config keys (Cycle 5.1 client foundation)
- `computers[]`: list of remote computer entries; each entry has:
  - `id`: unique identifier string (e.g. `"desktop"`)
  - `label`: human-friendly name (optional, default `""`)
  - `base_url`: http(s) URL of the agent API on the remote machine (e.g. `"http://127.0.0.1:7777"`)
  - `auth_token_ref`: CredentialManager lookup key for the bearer token — **never store the token in config**; put it in `.env` or `config/credentials.json`
  - `enabled`: `true` (default) — disabled computers are hidden from list output and cannot be targeted
  - `allowlists.commands[]`: list of command names permitted for remote execution (enforced client-side before any network call)
  - `connect_timeout`: connection timeout in seconds (default `5.0`)
  - `read_timeout`: read timeout in seconds (default `30.0`)
- Computer client code lives in `rex/computers/` (config, client, service).
- Docs: `docs/computers.md`

## Windows computer control CLI commands (Cycle 5.1 client foundation)
- `rex pc list` — list enabled computers
- `rex pc list --all` — list all computers including disabled
- `rex pc status --id <id>` — query agent for host info (hostname, OS, user, time)
- `rex pc run --id <id> --yes -- <command> [args]` — run an allowlisted command on the remote computer (explicit high-risk confirmation required)

Safety rules:
- `allowlists.commands` is enforced **client-side** before any network call.
- Auth tokens are resolved via `CredentialManager` from `auth_token_ref`; tokens are never logged.
- Disabled computers produce a clear error; unknown IDs produce a clear error.
- `rex pc run` requires explicit `--yes` confirmation before remote execution.
- All tests are offline (fake in-process HTTP server; no real network calls).

## Integration testing rules
- Integration tests must not require real network credentials.
- Use deterministic mocks/fixtures/fake transports for IMAP/SMTP/Twilio/ICS.
- When mocking HTTPS URL fetches, also mock `socket.getaddrinfo` (used by SSRF validation) so the test does not depend on DNS resolution.
- Add both success and failure-path tests for each backend adapter.
- Never log raw secrets (tokens, passwords, app passwords, OAuth refresh tokens).

## Testing
- Pytest configuration source-of-truth is `[tool.pytest.ini_options]` in `pyproject.toml`; do not reintroduce `pytest.ini`.
- Default pytest addopts do not include coverage flags. `pytest -q` works after a base install (`pip install -e .`) without pytest-cov.
- Coverage runs in CI only; coverage flags (`--cov=rex --cov-report=...`) are passed explicitly in `.github/workflows/ci.yml`.
- Canonical local test command: `pytest -q`.
- For local coverage: `pip install -e '.[dev]'` then `pytest -q --cov=rex --cov-report=term-missing`.
- Do not make default dev commands depend on optional plugins unless those plugins are in dev extras and documented, otherwise keep those flags CI-only.
- Tests must not write to tracked repo fixtures or files (for example under `data/`). Use `tmp_path` or temp copies; only commit fixture updates when intentional.

## Security Audit
- Run security checks with: `python scripts/security_audit.py`.
- Optional strict mode: `python scripts/security_audit.py --strict-markdown-secrets`.
- Default behavior intentionally skips Markdown fenced code blocks for merge-marker and secret checks to reduce false positives.
- In strict mode, Markdown fenced blocks are scanned for merge markers and secrets; use `--allowlist <file>` to exempt specific Markdown files when needed.
- Any heuristic change in `scripts/security_audit.py` must include corresponding updates in `tests/test_security_audit.py` with both true-positive and false-positive coverage.
- Do not add silent blind spots: if skipping new content classes, document rationale in code/comments and add explicit tests that prove intended detection is still preserved.

## CI Rules
- Do not add soft-pass CI patterns that mask failures (for example `|| echo` on required checks).
- Node/JavaScript CI jobs must not be added unless a real Node project exists (for example `package.json` at repo root or explicit subpath).
- Any Node job that is added must fail on real lint/test/build errors.
- CI and tests enforce clean-repo isolation after test runs (tracked files must remain unchanged; coverage artifacts are excluded).

## Lint preflight (must follow before pushing)
Before pushing any branch, run Ruff and Black on all changed Python files against the base branch:
```bash
BASE_REF="master"
git fetch origin "$BASE_REF"
files=$(git diff --name-only "origin/$BASE_REF...HEAD" -- '*.py')
echo "$files" | xargs ruff check --fix
echo "$files" | xargs ruff check
echo "$files" | xargs black
echo "$files" | xargs black --check --diff
```
Fix all reported issues before committing. Do not push code that fails Ruff or Black.

## Commit and PR title rules (CI enforced)
This repo enforces Conventional Commits via a commitlint GitHub Action. It lints both:
- Every commit message in the PR, and
- The PR title (often used as the squash-merge commit message)

If either is not in Conventional Commits format, CI fails.

Required format:
- `type(optional-scope): subject`

Common types to use:
- `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`, `build`, `perf`

Examples:
- `docs: add integration backlog and update CLAUDE rules`
- `feat(email): add IMAP backend adapter scaffold`
- `fix(calendar): handle empty ICS feed gracefully`
- `chore(ci): tighten commitlint rules`

Subject guidelines:
- Use imperative mood ("add", "fix", "update"), do not start with "Added" or "Adding"
- No trailing period
- Keep it short and specific

## Docs Consistency
- Integration docs for `email`, `calendar`, `messaging`, and `notifications` must include a top-level Implementation Status block (Beta, Stub, or Production-ready).
- When README Current Limitations changes, update the corresponding integration docs in the same PR.
- Avoid capability wording that overstates readiness relative to implementation status.

## Pydantic v2 Conventions
- Use Pydantic v2 style: `ConfigDict`, `field_serializer`, and `model_dump` or `model_dump_json`.
- Do not introduce deprecated v1 patterns such as class-based `Config` with `json_encoders` in new or modified models.
- If serialization behavior changes, add or update tests to lock expected output.

## Rules
1. Read before writing.
   - Inspect existing modules and patterns before adding new ones.
   - Do not invent filenames or APIs that do not exist in this repo.

2. Respect the config split.
   - Secrets in `.env` only.
   - Runtime config in `config/rex_config.json`.

3. Windows compatibility matters.
   - Avoid dependencies known to fail on Windows unless optional and guarded.

4. Keep integrations optional.
   - Email, calendar, SMS, MQTT, Home Assistant, and web search must degrade gracefully if not configured.

5. Do not add network exposure by default.
   - Anything that binds to a port must be authenticated and rate limited.
   - Prefer localhost binding unless explicitly configured otherwise.

## Working Agreements for Claude Code
- If requirements are ambiguous, propose a safe default and explain it briefly.
- Provide the full updated file when you modify code, not a partial diff.
- Keep outputs paste-ready and avoid non-printing Unicode characters.
- Use Conventional Commits for every commit and for the PR title (see "Commit and PR title rules").

## Code output rules (must follow)
- Do not provide truncated code. If you change a file, output the entire updated file unless the request explicitly asks for a diff.
- Do not use placeholders like "..." or "rest of file" or "omitted for brevity" in code.
- Do not claim something is implemented unless the code shown fully implements it.
- If a change spans multiple files, list every file you changed and output each file in full.
- If you cannot complete an implementation due to missing information, state exactly what is missing and provide the best safe partial implementation with clear TODO markers.
- Every PR must include a minimal verification step that can be run locally (command plus expected outcome).

## Workflow feedback loop (must follow)
If the reviewer (Codex or a human) changes Claude’s output due to a recurring mistake or wrong assumption, update CLAUDE.md in the same PR with a new or refined rule that would have prevented the issue. Keep the rule short and specific.

## Maintenance rules for CLAUDE.md (must follow)
When you change the repo, you must keep this file accurate.

Update CLAUDE.md in the same PR when any of the following change:
- New commands, scripts, or make targets are added or existing ones change
- Project structure changes (new top-level folders, new services, renamed modules)
- Dependencies or runtime requirements change (Python version, system deps, Docker images)
- New environment variables, secrets, or config files are introduced
- New integrations are added (Home Assistant, IFTTT, Cloudflare Tunnel, etc.)
- Any “Rules” or “Conventions” in this file become outdated

Do not update CLAUDE.md for:
- Pure refactors that do not change behavior, commands, or structure
- Comment-only or formatting-only changes

PR requirements:
- Every PR must include a short Verification section describing how you tested the change
- If CLAUDE.md was updated, mention it explicitly in the PR description
