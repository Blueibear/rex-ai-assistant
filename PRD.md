# PRD: Rex AI Assistant — Next Cycle

> **Codex/Ralph task selection rule**
> A "task" means one full User Story (US-###), not an individual checkbox line.
> Choose the first US-### that contains any unchecked acceptance criteria `[ ]`.
> Complete the full story in one iteration. If it cannot be completed in one iteration, split it first.
> Only mark acceptance criteria `[x]` when the full story is done and tests pass.

---

## Introduction

All 174 stories in the base PRD.md are complete. This PRD covers the next cycle of work:
resolving outstanding CODEX audit findings, fixing code-level bugs identified in a fresh
review, aligning dependency artifacts, implementing real integration backends (email, calendar,
SMS), expanding features (conversation history, streaming LLM, configurable STT), and cleaning
up technical debt across the repository.

Stories are ordered by dependency. Earlier stories must not depend on later ones.

---

## Goals

- Eliminate all P0/P1 CODEX audit issues (SEC-001, COR-001, QLT-001, DEP-001, DOC-001, DOC-002)
- Restore and enforce all quality gates (Ruff, Black, mypy) to zero violations
- Fix every confirmed code-level bug from the post-completion review
- Replace stub email, calendar, and SMS backends with real implementations
- Add conversation history persistence so sessions survive restarts
- Add LLM streaming and configurable Whisper language
- Keep the full test suite green and all integrations gracefully degradable

---

## Non-Goals

- No mobile or native desktop UI
- No multi-tenant hosting or cloud deployment pipeline
- No billing, usage metering, or third-party plugin marketplace
- No OAuth calendar backends (Google/Microsoft) in this cycle — ICS feed only
- No GPU-specific CI runners

---

# PHASE A — Security and Docker Hardening (SEC-001)

### US-175: Harden .dockerignore to exclude secrets and local state

**Description:** As an operator, I want the Docker build context to exclude secrets and
local runtime state so that `docker build` never captures `.env`, credentials, or
development artifacts.

**Acceptance Criteria:**
- [x] `.dockerignore` excludes: `.env`, `.env.*`, `venv/`, `.venv/`, `config/credentials.json`,
  `data/`, `logs/`, `transcripts/`, `Memory/`, `session_summaries/`, `backups/`, `*.log`,
  `*.bundle`, `*.egg-info/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`
- [x] `.dockerignore` excludes test artifacts: `tests/`, `coverage.json`, `coverage.txt`
- [x] `docker build .` succeeds after the change
- [x] Running `docker build .` in a directory containing a `.env` file does NOT include
  `.env` in the image (verify with `docker run --rm <image> ls /app/.env || echo "not found"`)
- [x] Typecheck passes

---

### US-176: Replace broad Dockerfile COPY with allowlist

**Description:** As an operator, I want the Dockerfile runtime stage to copy only
production-required files so that the resulting image is minimal and safe.

**Acceptance Criteria:**
- [x] Dockerfile runtime stage replaces `COPY . .` with explicit allowlist covering only:
  `rex/`, `rex_speak_api.py`, `rex_loop.py`, `voice_loop.py`, `pyproject.toml`,
  `config/rex_config.example.json`, `assets/`, and entry-point scripts
- [x] Image builds successfully: `docker build -t rex-test .` exits 0
- [x] `docker run --rm rex-test python -c "import rex"` exits 0
- [x] Image does not contain `.env`, `tests/`, `venv/`, or `Memory/` directories
- [x] Dockerfile comments document which mounts are expected at runtime (config, data)
- [x] Typecheck passes

---

# PHASE B — Code Quality Restoration (QLT-001)

### US-177: Restore Ruff lint compliance — import and unused-code violations

**Description:** As a developer, I want all Ruff import-order and unused-symbol violations
fixed so that the linter baseline is clean before enforcing it in CI.

**Acceptance Criteria:**
- [x] `ruff check rex/ --select I,F` exits 0 (import order + unused imports/variables)
- [x] No `noqa` suppressions added that were not already present
- [x] `pytest -q` exits 0 after changes (no regressions)
- [x] Typecheck passes

---

### US-178: Restore Ruff lint compliance — remaining rule violations

**Description:** As a developer, I want all remaining Ruff violations (beyond import order)
resolved so that `ruff check rex/` exits clean.

**Acceptance Criteria:**
- [x] `ruff check rex/` exits 0 with zero errors
- [x] `ruff check rex/ --statistics` shows 0 total
- [x] No existing `# noqa` comments were silently widened to suppress new categories
- [x] `pytest -q` exits 0
- [x] Typecheck passes

---

### US-179: Restore Black formatting compliance

**Description:** As a developer, I want the full package formatted with Black so that
`black --check` passes on every Python file.

**Acceptance Criteria:**
- [x] `black --check rex/` exits 0
- [x] `black --check *.py` exits 0 for all root-level Python files
- [x] No logic changes introduced — only whitespace/formatting
- [x] `pytest -q` exits 0
- [x] Typecheck passes

---

### US-180: Resolve mypy type errors — batch 1 (core package, highest-impact files)

**Description:** As a developer, I want the highest-impact mypy errors in `rex/` resolved
so that type coverage improves measurably.

**Acceptance Criteria:**
- [x] `mypy rex/assistant.py rex/config.py rex/llm_client.py rex/voice_loop.py` exits 0
- [x] No `type: ignore` comments added without an inline explanation
- [x] `pytest -q` exits 0
- [x] Typecheck passes

---

### US-181: Resolve mypy type errors — batch 2 (integrations and remaining files)

**Description:** As a developer, I want mypy to pass on the full `rex/` package so that
the codebase has a clean type baseline.

**Acceptance Criteria:**
- [x] `mypy rex/` exits 0 with zero errors
- [x] All `type: ignore` comments that remained from batch 1 are either resolved or
  documented with a specific reason comment
- [x] `pytest -q` exits 0
- [x] Typecheck passes

---

# PHASE C — Test Infrastructure Fixes (TST-001)

### US-182: Fix brittle repo-integrity tests

**Description:** As a developer, I want repo-integrity tests to capture a git-status
baseline at the start of the test session and compare against that baseline so that
pre-existing dirty files do not cause false failures.

**Acceptance Criteria:**
- [x] `tests/test_repo_integrity.py` captures `git status --porcelain` output before any
  test runs and stores it as the session baseline
- [x] `tests/test_repository_integrity.py` uses the same baseline approach
- [x] Running `pytest -q tests/test_repo_integrity.py tests/test_repository_integrity.py`
  exits 0 even when `requirements-gpu-cu124.txt` (or any other pre-existing tracked
  modification) is already dirty
- [x] Both files include a comment explaining the baseline approach
- [x] Typecheck passes

---

# PHASE D — Operations Script Fixes (OPS-001, OPS-002)

### US-183: Fix security audit script false positives

**Description:** As a developer, I want the security audit script to scan only tracked
source files and exclude generated caches so that its output is actionable rather than noisy.

**Acceptance Criteria:**
- [x] `scripts/security_audit.py` excludes `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`,
  `__pycache__/`, `venv/`, `.venv/`, `*.egg-info/`, `build/`, and `node_modules/`
- [x] Total scanned file count reported separately from excluded files
- [x] Script output categorizes findings by: source code vs documentation vs configuration
- [x] Running `python scripts/security_audit.py` reports fewer than 50 findings on a
  clean checkout (eliminating the 295 false-positive count)
- [x] Typecheck passes

---

### US-184: Rewrite deployment validation script

**Description:** As an operator, I want the deployment validation script to reflect the
current runtime model so that it gives accurate pass/fail verdicts.

**Acceptance Criteria:**
- [x] `scripts/validate_deployment.py` checks for `config/rex_config.json` existence and
  schema validity instead of `REX_ACTIVE_USER` environment variable
- [x] Script validates torch version against the range in `pyproject.toml` (`>=2.6.0,<2.9.0`)
  instead of expecting `2.5.x`
- [x] Script validates that all CLI entrypoints from `pyproject.toml` are importable
- [x] `python scripts/validate_deployment.py` exits 0 on a properly configured install
- [x] Score output accurately reflects the 7 checks (all 7 passing on valid install)
- [x] Typecheck passes

---

# PHASE E — Execution Surface Correctness (COR-001)

### US-185: Define authoritative executable tool catalog

**Description:** As a developer, I want one canonical list of tools that are truly
executable end-to-end so that the Planner, registry, and router all agree on scope.

**Acceptance Criteria:**
- [x] New file `rex/tool_catalog.py` defines `EXECUTABLE_TOOLS: frozenset[str]` containing
  exactly the tools with real handlers: `time_now`, `weather_now`, `web_search`,
  `send_email`, `calendar_create_event`, `home_assistant_call_service`
- [x] `rex/tool_registry.py` imports and validates against `EXECUTABLE_TOOLS` at registration time
- [x] `rex/planner.py` imports `EXECUTABLE_TOOLS` and limits plan generation to that set
- [x] `rex/tool_router.py` imports `EXECUTABLE_TOOLS` and raises `UnknownToolError` (not a
  generic exception) for any tool not in the catalog
- [x] `pytest -q tests/test_tool_registry.py tests/test_tool_router.py` exits 0
- [x] Typecheck passes

---

### US-186: Implement weather_now and web_search tool handlers

**Description:** As a user, I want weather and web search queries to return real results
when those integrations are configured so that Rex can answer factual questions.

**Acceptance Criteria:**
- [x] `rex/tool_router.py` `weather_now` handler calls the configured weather provider
  (existing `rex/tools/weather.py` or equivalent) and returns a formatted string
- [x] `rex/tool_router.py` `web_search` handler calls the configured search provider
  (existing search integration) and returns top-3 result summaries
- [x] Both handlers return a graceful `"[integration not configured]"` string when no
  API key is present (not an exception)
- [x] `pytest -q tests/test_tool_router.py` exits 0 with mocked provider responses
- [x] Typecheck passes

---

### US-187: Implement send_email and calendar_create_event tool handlers

**Description:** As a user, I want the Planner to be able to send email and create
calendar events when those integrations are configured.

**Acceptance Criteria:**
- [x] `rex/tool_router.py` `send_email` handler accepts `{to, subject, body}` and calls
  `EmailService.send()`, returning `"Email sent"` or a descriptive error string
- [x] `rex/tool_router.py` `calendar_create_event` handler accepts `{title, start, end}`
  and calls `CalendarService.create_event()`, returning confirmation or error string
- [x] Both handlers degrade gracefully when the backend is not configured
- [x] `pytest -q tests/test_tool_router.py` exits 0
- [x] Typecheck passes

---

### US-188: Add planner-to-router end-to-end integration tests

**Description:** As a developer, I want integration tests proving that every tool the
Planner can emit is executable through the router so that COR-001 cannot regress.

**Acceptance Criteria:**
- [x] New file `tests/test_planner_tool_e2e.py` tests each tool in `EXECUTABLE_TOOLS`
- [x] Each test: generates a minimal plan containing that tool, executes it through
  `execute_tool()`, asserts the result is a non-empty string (not an exception)
- [x] All tests use mocked external services (no real API calls)
- [x] `pytest -q tests/test_planner_tool_e2e.py` exits 0
- [x] Typecheck passes

---

# PHASE F — Documentation Truth (DOC-001, DOC-002, DOC-003, ARC-001)

### US-189: Align README runtime configuration section

**Description:** As a user, I want the README to describe the actual config system
(JSON runtime config + secrets-only .env) so that setup instructions are not misleading.

**Acceptance Criteria:**
- [x] README no longer presents large env-var configuration tables as the primary setup method
- [x] README clearly states: secrets go in `.env`, runtime settings go in
  `config/rex_config.json`, and links to `CONFIGURATION.md` for the full reference
- [x] `.env.example` is referenced in the README and its role is described accurately
- [x] Existing Quick Start section steps remain accurate after the change
- [x] Typecheck passes

---

### US-190: Rewrite Windows quickstart with correct entrypoints

**Description:** As a Windows user, I want the Windows setup guide to describe the real
startup commands for each runtime mode so that I can launch Rex successfully.

**Acceptance Criteria:**
- [x] `README.windows.md` describes four distinct runtime modes with their correct commands:
  text chat (`python -m rex`), voice loop (`python rex_loop.py`),
  dashboard (`python run_gui.py`), TTS API (`python rex_speak_api.py`)
- [x] All PowerShell activation and launch commands are tested as syntactically correct
- [x] Guide no longer instructs users to configure runtime behavior via environment variables
- [x] Guide references `config/rex_config.example.json` for runtime configuration
- [x] Typecheck passes

---

### US-191: Archive and correct stale architecture and status documents

**Description:** As a developer, I want stale docs that claim "production-ready" status
or describe retired architecture to be clearly archived or corrected so that no document
misleads contributors.

**Acceptance Criteria:**
- [x] Any document claiming "production-ready" status is updated to reflect the actual
  state or moved to `docs/archive/` with an `ARCHIVED:` prefix in its title
- [x] Documents referencing the retired OpenClaw Python package import architecture are
  updated to describe the current HTTP integration approach
- [x] `docs/claude/INTEGRATIONS_STATUS.md` accurately reflects which integrations are real
  vs stub (email, calendar, SMS)
- [x] Typecheck passes

---

### US-192: Consolidate to one canonical voice loop entry point

**Description:** As a developer, I want the voice loop duplication between root-level and
`rex/` package resolved so that there is one documented, authoritative startup path.

**Acceptance Criteria:**
- [x] `CLAUDE.md` documents which file is the canonical voice loop entry point and why
  the second exists (or it is removed if unused)
- [x] `rex_loop.py` (root) explicitly imports from the canonical module and does not
  duplicate business logic
- [x] Both `voice_loop.py` files have a header comment explaining their relationship
- [x] `pytest -q` exits 0
- [x] Typecheck passes

---

# PHASE G — Dependency Alignment (DEP-001)

### US-193: Define and document the canonical runtime matrix

**Description:** As a developer, I want one authoritative version matrix for Python,
PyTorch, and CUDA targets so that requirements files, pyproject.toml, the Dockerfile,
and the validation script all agree.

**Acceptance Criteria:**
- [x] `docs/DEPENDENCIES.md` (new or updated) documents the canonical matrix:
  Python 3.10–3.13, torch 2.6.x–2.8.x (CPU), cu118 variant, cu124 variant
- [x] `requirements-cpu.txt` does not pin a CUDA variant of torch
- [x] `pyproject.toml` optional ML extras match the documented matrix
- [x] `Dockerfile` torch version falls within the documented matrix
- [x] `scripts/validate_deployment.py` (from US-184) validates against the documented matrix
- [x] Typecheck passes

---

# PHASE H — Code Bug Fixes (from code review)

### US-194: Thread-safe TTS engine in rex_speak_api.py

**Description:** As an operator, I want the TTS engine to be protected by a threading lock
so that concurrent HTTP requests to `/speak` cannot cause race conditions or corrupt state.

**Acceptance Criteria:**
- [x] `rex_speak_api.py` introduces `_tts_lock = threading.Lock()` at module level
- [x] All access to `_TTS_ENGINE` (read and write) is wrapped in `with _tts_lock:`
- [x] The lock is acquired before initialization check and released after synthesis completes
- [x] A test in `tests/test_speak_api.py` (or new file) fires 5 concurrent `/speak` requests
  using `threading.Thread` and asserts all return 200 with non-empty audio
- [x] Typecheck passes

---

### US-195: Fix _followup_injected race condition in assistant.py

**Description:** As a developer, I want the followup injection flag in `assistant.py` to
use an `asyncio.Lock` so that concurrent `generate_reply` calls cannot inject followup
context twice.

**Acceptance Criteria:**
- [x] `Assistant` replaces `self._followup_injected: bool` with
  `self._followup_lock: asyncio.Lock` initialized in `__init__`
- [x] The injection block is wrapped in `async with self._followup_lock:`
- [x] A test simulates two concurrent `generate_reply` calls and asserts followup context
  is injected at most once across both calls
- [x] `pytest -q tests/test_assistant.py` exits 0
- [x] Typecheck passes

---

### US-196: Fix inconsistent temp file cleanup in voice_loop.py

**Description:** As a developer, I want all temporary audio files created during
speech processing to be cleaned up reliably so that orphaned files do not accumulate.

**Acceptance Criteria:**
- [x] All `tempfile.NamedTemporaryFile` and manual temp-path usages in `rex/voice_loop.py`
  and root `voice_loop.py` are wrapped in `try/finally` blocks that call `os.unlink()`
- [x] Cleanup catches and logs `OSError` / `PermissionError` instead of raising
- [x] A test creates a voice processing cycle with a mock audio source and asserts the
  temp directory has no leftover `.wav` files after the call
- [x] `pytest -q tests/test_voice_loop.py` exits 0
- [x] Typecheck passes

---

### US-197: Process OpenAI tool_calls in LLM client

**Description:** As a developer, I want the OpenAI strategy to correctly handle
`tool_calls` in responses so that function-calling is not silently ignored.

**Acceptance Criteria:**
- [x] `rex/llm_client.py` `OpenAIStrategy.generate()` checks `message.tool_calls` and,
  when present, serializes them into the response string as structured JSON tool call data
- [x] A test mocks an OpenAI response with `tool_calls` and asserts the returned
  string contains the tool name and arguments
- [x] No existing tests regress
- [x] Typecheck passes

---

### US-198: Fix Ollama error message taxonomy

**Description:** As a developer, I want Ollama connection errors to be distinguished
from missing-model errors so that users see accurate troubleshooting messages.

**Acceptance Criteria:**
- [x] `OllamaStrategy` catches `httpx.ConnectError` / `ConnectionRefusedError` and returns
  `"[Ollama: connection failed — is Ollama running?]"`
- [x] `OllamaStrategy` catches 404 model-not-found responses and returns
  `"[Ollama: model '{model}' not found — run: ollama pull {model}]"`
- [x] All other errors return `"[Ollama: unexpected error: {detail}]"`
- [x] Tests cover all three error paths with mocked HTTP responses
- [x] Typecheck passes

---

### US-199: Fix sentence splitting for abbreviations in TTS pipeline

**Description:** As a user, I want TTS sentence splitting to not break mid-sentence on
abbreviations like "Dr.", "Mr.", "e.g.", and "etc." so that speech sounds natural.

**Acceptance Criteria:**
- [x] `rex/voice_loop.py` sentence splitter uses an abbreviation-aware approach
  (either an allowlist of common titles/abbreviations or NLTK `sent_tokenize` if available,
  with regex fallback)
- [x] "Dr. Smith said the treatment works." is treated as a single sentence
- [x] "She said it was great. He agreed." is correctly split into two sentences
- [x] "e.g. this example." is treated as one sentence
- [x] Existing TTS pipeline tests pass
- [x] Typecheck passes

---

### US-200: Add request body size limit to rex_speak_api.py

**Description:** As an operator, I want the `/speak` endpoint to reject request bodies
over a configurable size limit so that the server cannot be exhausted by oversized payloads.

**Acceptance Criteria:**
- [x] `rex_speak_api.py` reads `MAX_REQUEST_BYTES` from config (default: 64 KB)
- [x] Requests where `Content-Length` exceeds `MAX_REQUEST_BYTES` are rejected with 413
  before the body is read
- [x] Requests without `Content-Length` are read with a stream cap at `MAX_REQUEST_BYTES`
- [x] A test sends a request body of 100 KB and asserts the response is 413
- [x] Typecheck passes

---

### US-201: Fix suppressed JSON errors in identity.py

**Description:** As a developer, I want session file JSON parse errors to be logged as
warnings so that corrupted session files are detectable and diagnosable.

**Acceptance Criteria:**
- [x] `rex/identity.py` `_load_session()` catches `json.JSONDecodeError` and logs a
  `logger.warning(f"Corrupted session file {path}, resetting: {e}")` before returning `{}`
- [x] The function no longer silently swallows parse errors
- [x] A test writes a malformed JSON session file and asserts: the function returns `{}`
  and a warning is logged
- [x] Typecheck passes

---

# PHASE I — Conversation History Persistence

### US-202: Create SQLite conversation history schema and HistoryStore class

**Description:** As a developer, I want a `HistoryStore` backed by SQLite so that
conversation turns can be persisted and retrieved across restarts.

**Acceptance Criteria:**
- [x] New file `rex/history_store.py` defines `HistoryStore` with:
  - `__init__(self, db_path: Path)` — creates/migrates the DB on first call
  - `save_turn(user_id: str, role: str, content: str, timestamp: datetime) -> None`
  - `load_history(user_id: str, limit: int = 50) -> list[dict]`
  - `prune(user_id: str, keep_days: int = 30) -> int` — returns rows deleted
- [x] Schema uses a single `turns` table: `(id, user_id, role, content, timestamp)`
- [x] DB is created at `data/history.db` by default, path is configurable
- [x] `pytest -q tests/test_history_store.py` exits 0 (new test file covers CRUD + prune)
- [x] Typecheck passes

---

### US-203: Wire HistoryStore into assistant.py

**Description:** As a user, I want conversation history to survive assistant restarts so
that Rex can reference previous exchanges without the session being lost.

**Acceptance Criteria:**
- [x] `Assistant.__init__` instantiates `HistoryStore` if `config.persist_history` is True
  (default: True)
- [x] `Assistant.generate_reply()` calls `history_store.save_turn()` for each user prompt
  and assistant response
- [x] `Assistant.__init__` preloads the last 50 turns from `HistoryStore` into the in-memory
  history on startup
- [x] `AppConfig` gains a `persist_history: bool = True` field
- [x] `pytest -q tests/test_assistant.py` exits 0
- [x] Typecheck passes

---

### US-204: Add history rotation/pruning scheduled task

**Description:** As an operator, I want history older than a configurable retention
window to be pruned automatically so that the database does not grow unbounded.

**Acceptance Criteria:**
- [x] `AppConfig` gains `history_retention_days: int = 30`
- [x] `rex/history_store.py` `prune()` is called by the scheduler (or a startup hook)
  once per day
- [x] Pruning is idempotent — running twice produces the same result as running once
- [x] A test asserts that turns older than retention window are deleted and recent turns
  are preserved
- [x] Typecheck passes

---

# PHASE J — Integration Backends

### US-205: Add transport-layer interfaces for email, calendar, and SMS

**Description:** As a developer, I want explicit protocol/ABC interfaces for each
integration backend so that real and stub implementations are interchangeable.

**Acceptance Criteria:**
- [x] `rex/integrations/email/backends/base.py` defines `EmailBackend` ABC with:
  `fetch_unread(limit: int) -> list[dict]` and `send(to, subject, body) -> None`
- [x] `rex/integrations/calendar/backends/base.py` defines `CalendarBackend` ABC with:
  `get_upcoming(days: int) -> list[dict]` and `create_event(title, start, end) -> dict`
- [x] `rex/integrations/messaging/backends/base.py` defines `SMSBackend` ABC with:
  `send(to, body) -> None` and `receive() -> list[dict]`
- [x] Existing mock/stub implementations are refactored to implement these interfaces
- [x] `pytest -q` exits 0 (no regressions)
- [x] Typecheck passes

---

### US-206: Implement IMAP read backend

**Description:** As a user, I want Rex to fetch real email from an IMAP server so that
the "read my email" command returns live inbox contents.

**Acceptance Criteria:**
- [x] New file `rex/integrations/email/backends/imap_smtp.py` defines `IMAPBackend`
  implementing `EmailBackend.fetch_unread()`
- [x] Uses stdlib `imaplib.IMAP4_SSL` with configurable host, port, and SSL flag
- [x] Connection timeout is enforced (default: 10 s)
- [x] On auth failure, raises a descriptive `EmailAuthError` (not a raw exception)
- [x] `tests/test_email_backend_imap_smtp.py` tests happy-path and auth-failure cases
  using `unittest.mock` on the socket layer (no live network calls)
- [x] `pytest -q tests/test_email_backend_imap_smtp.py` exits 0
- [x] Typecheck passes

---

### US-207: Implement SMTP send backend

**Description:** As a user, I want Rex to send real email via SMTP so that "send email"
commands deliver to the actual recipient.

**Acceptance Criteria:**
- [x] `IMAPSMTPBackend` (same file as US-206) implements `EmailBackend.send()` using
  stdlib `smtplib.SMTP` with STARTTLS or `smtplib.SMTP_SSL`
- [x] Credentials are loaded via `CredentialManager` using the account's `credential_ref`
- [x] Sensitive data (password) is never logged
- [x] Tests cover: successful send, auth failure, TLS failure, and timeout — all with mocks
- [x] `pytest -q tests/test_email_backend_imap_smtp.py` exits 0
- [x] Typecheck passes

---

### US-208: Add multi-account email config and routing

**Description:** As a user, I want to configure multiple email accounts and have Rex route
reads/sends to the correct account so that work and personal email are separate.

**Acceptance Criteria:**
- [x] `AppConfig` (via `rex_config.json`) supports:
  `email.accounts[]` (list of account objects with `id`, `address`, `imap`, `smtp`,
  `credential_ref`) and `email.default_account_id`
- [x] `EmailService` accepts optional `account_id` on `fetch_unread()` and `send()`;
  falls back to `default_account_id` when omitted
- [x] Invalid `account_id` raises `ValueError` with an actionable message
- [x] Backward-compatible when only a single legacy account config is present
- [x] `pytest -q tests/test_email_multi_account.py` exits 0 (new test file)
- [x] Typecheck passes

---

### US-209: Wire notification email channel to real send backend

**Description:** As a user, I want urgent and digest notifications to be delivered via
the real SMTP backend so that I receive email alerts.

**Acceptance Criteria:**
- [x] `rex/notification.py` `_send_to_email()` replaces the `"Would send."` log with a
  real call to `EmailService.send()`
- [x] Digest flush also dispatches through `EmailService`
- [x] On send failure, notification is marked failed (not silently dropped) and logged
- [x] `pytest -q tests/test_notification_email_delivery.py` exits 0 (new test file)
- [x] Typecheck passes

---

### US-210: Implement ICS calendar read-only feed backend

**Description:** As a user, I want Rex to read events from an ICS file or URL feed so
that the "upcoming events" command returns real calendar data.

**Acceptance Criteria:**
- [x] New file `rex/integrations/calendar/backends/ics_feed.py` defines `ICSFeedBackend`
  implementing `CalendarBackend.get_upcoming()`
- [x] Accepts a local file path or HTTP URL as the feed source
- [x] Normalizes event timezones to UTC internally
- [x] Handles malformed VEVENT blocks gracefully (logs warning, skips entry)
- [x] Uses stdlib only (no `icalendar` package) for base parsing; falls back to `icalendar`
  if installed
- [x] `tests/test_calendar_ics_backend.py` uses fixture `.ics` files; no live HTTP
- [x] `pytest -q tests/test_calendar_ics_backend.py` exits 0
- [x] Typecheck passes

---

### US-211: Implement Twilio SMS send adapter

**Description:** As a user, I want Rex to send real SMS messages via Twilio so that the
"send SMS" command delivers to the recipient's phone.

**Acceptance Criteria:**
- [x] New file `rex/integrations/messaging/backends/twilio_sms.py` defines `TwilioSMSBackend`
  implementing `SMSBackend.send()`
- [x] Uses the `twilio` optional extra; imports are guarded with a helpful error if not installed
- [x] Credentials (`account_sid`, `auth_token`, `from_number`) loaded via `CredentialManager`
- [x] On 4xx response: raises `SMSSendError` with Twilio error code
- [x] On network timeout: raises `SMSSendError` with timeout detail
- [x] No secrets logged at any log level
- [x] `pytest -q tests/test_twilio_sms_backend.py` exits 0 with mocked Twilio client
- [x] Typecheck passes

---

### US-212: Add offline integration test harnesses

**Description:** As a developer, I want fake IMAP, SMTP, and Twilio transports available
as test fixtures so that integration tests run fully offline with no credentials.

**Acceptance Criteria:**
- [ ] `tests/helpers/fake_imap.py` provides a `FakeIMAP4SSL` class that behaves like
  `imaplib.IMAP4_SSL` for use in tests
- [ ] `tests/helpers/fake_smtp.py` provides a `FakeSMTP` class for both `SMTP` and `SMTP_SSL`
- [ ] `tests/helpers/fake_twilio.py` provides a fake Twilio `Client` fixture
- [ ] All existing `tests/test_email_backend_imap_smtp.py` and `tests/test_twilio_sms_backend.py`
  tests switch to using these helpers
- [ ] `pytest -q tests/test_email_backend_imap_smtp.py tests/test_twilio_sms_backend.py`
  exits 0 with no live network access
- [ ] Typecheck passes

---

# PHASE K — Feature Expansions

### US-213: Add configurable Whisper language

**Description:** As a user, I want to configure the Whisper transcription language so
that non-English speakers receive accurate transcriptions.

**Acceptance Criteria:**
- [ ] `AppConfig` gains `whisper_language: str = "en"` (None = auto-detect)
- [ ] `SpeechToText` passes `language=config.whisper_language` to `whisper.transcribe()`
- [ ] Setting `whisper_language = null` in config enables Whisper auto-detection
- [ ] `rex doctor` output includes the current `whisper_language` value
- [ ] `pytest -q tests/test_speech_to_text.py` exits 0
- [ ] Typecheck passes

---

### US-214: Add WAV audio format validation

**Description:** As a developer, I want audio files passed to the STT pipeline validated
as WAV before transcription so that malformed files fail with a clear error rather than
a cryptic exception.

**Acceptance Criteria:**
- [ ] `SpeechToText.transcribe()` validates the first 4 bytes of the audio buffer equal
  `b"RIFF"` (WAV magic bytes) before passing to Whisper
- [ ] Non-WAV input raises `AudioFormatError("Expected WAV, got {detected_format}")`
- [ ] The exception is caught in the voice loop and logged; the loop re-arms without crashing
- [ ] A test passes a fake MP3 header and asserts `AudioFormatError` is raised
- [ ] Typecheck passes

---

### US-215: Add audio device validation at startup

**Description:** As a user, I want Rex to validate audio input/output device availability
at startup so that misconfigured devices produce a clear error before the voice loop begins.

**Acceptance Criteria:**
- [ ] During voice loop initialization, the configured input device index is validated
  against `sounddevice.query_devices()`
- [ ] Invalid device index raises `AudioDeviceError(f"Input device {idx} not found. "
  f"Available: {available_list}")` before the wake word listener starts
- [ ] `rex doctor` includes an audio device check and prints available device names
- [ ] A test mocks `sounddevice.query_devices()` to simulate a missing device and asserts
  the correct error is raised
- [ ] Typecheck passes

---

### US-216: Add session expiration to identity.py

**Description:** As a developer, I want session files older than a configurable TTL to be
treated as expired so that stale user selections do not persist indefinitely.

**Acceptance Criteria:**
- [ ] `rex/identity.py` `_load_session()` reads the session file `mtime` and rejects it
  if older than `SESSION_TTL_HOURS` (default: 8)
- [ ] Expired sessions are deleted and `{}` is returned
- [ ] `AppConfig` gains `session_ttl_hours: int = 8`
- [ ] A test writes a session file with an artificially old mtime and asserts the
  function returns `{}` and deletes the file
- [ ] Typecheck passes

---

### US-217: Add LLM streaming interface and OpenAI streaming implementation

**Description:** As a user, I want the OpenAI LLM provider to stream tokens so that the
first words of a response appear faster while the rest is being generated.

**Acceptance Criteria:**
- [ ] `LanguageModelStrategy` protocol gains `stream(messages, **kwargs) -> Iterator[str]`
  method (default implementation raises `NotImplementedError`)
- [ ] `OpenAIStrategy.stream()` uses `stream=True` and yields token deltas as strings
- [ ] The voice loop detects streaming availability and feeds tokens to the TTS sentence
  buffer as they arrive (sentence-boundary flush unchanged)
- [ ] `EchoStrategy.stream()` yields the prompt text word by word (for testing)
- [ ] `pytest -q tests/test_llm_client.py` exits 0
- [ ] Typecheck passes

---

### US-218: Add streaming for Anthropic and Ollama providers

**Description:** As a user, I want Anthropic and Ollama providers to also stream tokens
so that all supported backends benefit from lower perceived latency.

**Acceptance Criteria:**
- [ ] `AnthropicStrategy.stream()` uses the Anthropic streaming API and yields string deltas
- [ ] `OllamaStrategy.stream()` uses Ollama's streaming endpoint and yields string deltas
- [ ] Both implementations handle stream interruptions gracefully (log warning, return
  what was collected)
- [ ] `pytest -q tests/test_llm_client.py` exits 0
- [ ] Typecheck passes

---

# PHASE L — Verification and Quality Gate Enforcement

### US-219: Add pre-commit config for Ruff and Black

**Description:** As a developer, I want a `.pre-commit-config.yaml` that runs Ruff and
Black on staged files so that lint and format regressions are caught before they reach CI.

**Acceptance Criteria:**
- [ ] `.pre-commit-config.yaml` exists at the repo root with hooks for:
  `ruff` (autofix: true) and `black`
- [ ] `pre-commit run --all-files` exits 0 on a clean checkout
- [ ] `CONTRIBUTING.md` (or `CLAUDE.md`) documents: `pip install pre-commit && pre-commit install`
- [ ] Typecheck passes

---

### US-220: Verify full test suite is green and coverage meets threshold

**Description:** As a developer, I want the complete test suite to pass with coverage at
or above the 75% threshold so that this cycle is provably complete.

**Acceptance Criteria:**
- [ ] `pytest -q` exits 0 with no failures or errors
- [ ] `pytest --cov=rex --cov-report=term-missing` reports overall coverage >= 75%
- [ ] No test is marked `xfail` that was previously passing
- [ ] All new test files from this cycle are included in the run
- [ ] Typecheck passes

---

## Non-Goals (This Cycle)

- No OAuth-based calendar backends (Google Calendar, Microsoft 365) — ICS feed is sufficient
- No streaming UI in the web dashboard (streaming goes to voice pipeline only)
- No mobile push notification channel
- No multi-user role management or RBAC
- No Kubernetes or container orchestration
- No audio level normalization or AGC (automatic gain control)
- No runtime hot-swap of audio devices
- No token counting pre-validation for LLM inputs (tracked for next cycle)

---

## Technical Reference

**Quality gate commands (run before marking any story complete):**
```bash
ruff check rex/
black --check rex/ *.py
mypy rex/
pytest -q
```

**Validation commands:**
```bash
python scripts/doctor.py
python scripts/validate_deployment.py
```

**Story ordering note:**
- US-175–US-176 (Docker) have no dependencies
- US-177–US-181 (quality) have no dependencies on each other; run in order
- US-182 (test fix) has no dependencies
- US-183–US-184 (ops scripts) have no dependencies
- US-185 must precede US-186, US-187, US-188
- US-189–US-192 (docs) have no dependencies
- US-193 (deps) has no dependencies
- US-194–US-201 (bug fixes) have no dependencies on each other
- US-202 must precede US-203, US-204
- US-205 must precede US-206, US-207, US-208, US-209, US-210, US-211, US-212
- US-217 must precede US-218
- US-219 has no dependencies
- US-220 must come last
