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
- [ ] Dockerfile runtime stage replaces `COPY . .` with explicit allowlist covering only:
  `rex/`, `rex_speak_api.py`, `rex_loop.py`, `voice_loop.py`, `pyproject.toml`,
  `config/rex_config.example.json`, `assets/`, and entry-point scripts
- [ ] Image builds successfully: `docker build -t rex-test .` exits 0
- [ ] `docker run --rm rex-test python -c "import rex"` exits 0
- [ ] Image does not contain `.env`, `tests/`, `venv/`, or `Memory/` directories
- [ ] Dockerfile comments document which mounts are expected at runtime (config, data)
- [ ] Typecheck passes

---

# PHASE B — Code Quality Restoration (QLT-001)

### US-177: Restore Ruff lint compliance — import and unused-code violations

**Description:** As a developer, I want all Ruff import-order and unused-symbol violations
fixed so that the linter baseline is clean before enforcing it in CI.

**Acceptance Criteria:**
- [ ] `ruff check rex/ --select I,F` exits 0 (import order + unused imports/variables)
- [ ] No `noqa` suppressions added that were not already present
- [ ] `pytest -q` exits 0 after changes (no regressions)
- [ ] Typecheck passes

---

### US-178: Restore Ruff lint compliance — remaining rule violations

**Description:** As a developer, I want all remaining Ruff violations (beyond import order)
resolved so that `ruff check rex/` exits clean.

**Acceptance Criteria:**
- [ ] `ruff check rex/` exits 0 with zero errors
- [ ] `ruff check rex/ --statistics` shows 0 total
- [ ] No existing `# noqa` comments were silently widened to suppress new categories
- [ ] `pytest -q` exits 0
- [ ] Typecheck passes

---

### US-179: Restore Black formatting compliance

**Description:** As a developer, I want the full package formatted with Black so that
`black --check` passes on every Python file.

**Acceptance Criteria:**
- [ ] `black --check rex/` exits 0
- [ ] `black --check *.py` exits 0 for all root-level Python files
- [ ] No logic changes introduced — only whitespace/formatting
- [ ] `pytest -q` exits 0
- [ ] Typecheck passes

---

### US-180: Resolve mypy type errors — batch 1 (core package, highest-impact files)

**Description:** As a developer, I want the highest-impact mypy errors in `rex/` resolved
so that type coverage improves measurably.

**Acceptance Criteria:**
- [ ] `mypy rex/assistant.py rex/config.py rex/llm_client.py rex/voice_loop.py` exits 0
- [ ] No `type: ignore` comments added without an inline explanation
- [ ] `pytest -q` exits 0
- [ ] Typecheck passes

---

### US-181: Resolve mypy type errors — batch 2 (integrations and remaining files)

**Description:** As a developer, I want mypy to pass on the full `rex/` package so that
the codebase has a clean type baseline.

**Acceptance Criteria:**
- [ ] `mypy rex/` exits 0 with zero errors
- [ ] All `type: ignore` comments that remained from batch 1 are either resolved or
  documented with a specific reason comment
- [ ] `pytest -q` exits 0
- [ ] Typecheck passes

---

# PHASE C — Test Infrastructure Fixes (TST-001)

### US-182: Fix brittle repo-integrity tests

**Description:** As a developer, I want repo-integrity tests to capture a git-status
baseline at the start of the test session and compare against that baseline so that
pre-existing dirty files do not cause false failures.

**Acceptance Criteria:**
- [ ] `tests/test_repo_integrity.py` captures `git status --porcelain` output before any
  test runs and stores it as the session baseline
- [ ] `tests/test_repository_integrity.py` uses the same baseline approach
- [ ] Running `pytest -q tests/test_repo_integrity.py tests/test_repository_integrity.py`
  exits 0 even when `requirements-gpu-cu124.txt` (or any other pre-existing tracked
  modification) is already dirty
- [ ] Both files include a comment explaining the baseline approach
- [ ] Typecheck passes

---

# PHASE D — Operations Script Fixes (OPS-001, OPS-002)

### US-183: Fix security audit script false positives

**Description:** As a developer, I want the security audit script to scan only tracked
source files and exclude generated caches so that its output is actionable rather than noisy.

**Acceptance Criteria:**
- [ ] `scripts/security_audit.py` excludes `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`,
  `__pycache__/`, `venv/`, `.venv/`, `*.egg-info/`, `build/`, and `node_modules/`
- [ ] Total scanned file count reported separately from excluded files
- [ ] Script output categorizes findings by: source code vs documentation vs configuration
- [ ] Running `python scripts/security_audit.py` reports fewer than 50 findings on a
  clean checkout (eliminating the 295 false-positive count)
- [ ] Typecheck passes

---

### US-184: Rewrite deployment validation script

**Description:** As an operator, I want the deployment validation script to reflect the
current runtime model so that it gives accurate pass/fail verdicts.

**Acceptance Criteria:**
- [ ] `scripts/validate_deployment.py` checks for `config/rex_config.json` existence and
  schema validity instead of `REX_ACTIVE_USER` environment variable
- [ ] Script validates torch version against the range in `pyproject.toml` (`>=2.6.0,<2.9.0`)
  instead of expecting `2.5.x`
- [ ] Script validates that all CLI entrypoints from `pyproject.toml` are importable
- [ ] `python scripts/validate_deployment.py` exits 0 on a properly configured install
- [ ] Score output accurately reflects the 7 checks (all 7 passing on valid install)
- [ ] Typecheck passes

---

# PHASE E — Execution Surface Correctness (COR-001)

### US-185: Define authoritative executable tool catalog

**Description:** As a developer, I want one canonical list of tools that are truly
executable end-to-end so that the Planner, registry, and router all agree on scope.

**Acceptance Criteria:**
- [ ] New file `rex/tool_catalog.py` defines `EXECUTABLE_TOOLS: frozenset[str]` containing
  exactly the tools with real handlers: `time_now`, `weather_now`, `web_search`,
  `send_email`, `calendar_create_event`, `home_assistant_call_service`
- [ ] `rex/tool_registry.py` imports and validates against `EXECUTABLE_TOOLS` at registration time
- [ ] `rex/planner.py` imports `EXECUTABLE_TOOLS` and limits plan generation to that set
- [ ] `rex/tool_router.py` imports `EXECUTABLE_TOOLS` and raises `UnknownToolError` (not a
  generic exception) for any tool not in the catalog
- [ ] `pytest -q tests/test_tool_registry.py tests/test_tool_router.py` exits 0
- [ ] Typecheck passes

---

### US-186: Implement weather_now and web_search tool handlers

**Description:** As a user, I want weather and web search queries to return real results
when those integrations are configured so that Rex can answer factual questions.

**Acceptance Criteria:**
- [ ] `rex/tool_router.py` `weather_now` handler calls the configured weather provider
  (existing `rex/tools/weather.py` or equivalent) and returns a formatted string
- [ ] `rex/tool_router.py` `web_search` handler calls the configured search provider
  (existing search integration) and returns top-3 result summaries
- [ ] Both handlers return a graceful `"[integration not configured]"` string when no
  API key is present (not an exception)
- [ ] `pytest -q tests/test_tool_router.py` exits 0 with mocked provider responses
- [ ] Typecheck passes

---

### US-187: Implement send_email and calendar_create_event tool handlers

**Description:** As a user, I want the Planner to be able to send email and create
calendar events when those integrations are configured.

**Acceptance Criteria:**
- [ ] `rex/tool_router.py` `send_email` handler accepts `{to, subject, body}` and calls
  `EmailService.send()`, returning `"Email sent"` or a descriptive error string
- [ ] `rex/tool_router.py` `calendar_create_event` handler accepts `{title, start, end}`
  and calls `CalendarService.create_event()`, returning confirmation or error string
- [ ] Both handlers degrade gracefully when the backend is not configured
- [ ] `pytest -q tests/test_tool_router.py` exits 0
- [ ] Typecheck passes

---

### US-188: Add planner-to-router end-to-end integration tests

**Description:** As a developer, I want integration tests proving that every tool the
Planner can emit is executable through the router so that COR-001 cannot regress.

**Acceptance Criteria:**
- [ ] New file `tests/test_planner_tool_e2e.py` tests each tool in `EXECUTABLE_TOOLS`
- [ ] Each test: generates a minimal plan containing that tool, executes it through
  `execute_tool()`, asserts the result is a non-empty string (not an exception)
- [ ] All tests use mocked external services (no real API calls)
- [ ] `pytest -q tests/test_planner_tool_e2e.py` exits 0
- [ ] Typecheck passes

---

# PHASE F — Documentation Truth (DOC-001, DOC-002, DOC-003, ARC-001)

### US-189: Align README runtime configuration section

**Description:** As a user, I want the README to describe the actual config system
(JSON runtime config + secrets-only .env) so that setup instructions are not misleading.

**Acceptance Criteria:**
- [ ] README no longer presents large env-var configuration tables as the primary setup method
- [ ] README clearly states: secrets go in `.env`, runtime settings go in
  `config/rex_config.json`, and links to `CONFIGURATION.md` for the full reference
- [ ] `.env.example` is referenced in the README and its role is described accurately
- [ ] Existing Quick Start section steps remain accurate after the change
- [ ] Typecheck passes

---

### US-190: Rewrite Windows quickstart with correct entrypoints

**Description:** As a Windows user, I want the Windows setup guide to describe the real
startup commands for each runtime mode so that I can launch Rex successfully.

**Acceptance Criteria:**
- [ ] `README.windows.md` describes four distinct runtime modes with their correct commands:
  text chat (`python -m rex`), voice loop (`python rex_loop.py`),
  dashboard (`python run_gui.py`), TTS API (`python rex_speak_api.py`)
- [ ] All PowerShell activation and launch commands are tested as syntactically correct
- [ ] Guide no longer instructs users to configure runtime behavior via environment variables
- [ ] Guide references `config/rex_config.example.json` for runtime configuration
- [ ] Typecheck passes

---

### US-191: Archive and correct stale architecture and status documents

**Description:** As a developer, I want stale docs that claim "production-ready" status
or describe retired architecture to be clearly archived or corrected so that no document
misleads contributors.

**Acceptance Criteria:**
- [ ] Any document claiming "production-ready" status is updated to reflect the actual
  state or moved to `docs/archive/` with an `ARCHIVED:` prefix in its title
- [ ] Documents referencing the retired OpenClaw Python package import architecture are
  updated to describe the current HTTP integration approach
- [ ] `docs/claude/INTEGRATIONS_STATUS.md` accurately reflects which integrations are real
  vs stub (email, calendar, SMS)
- [ ] Typecheck passes

---

### US-192: Consolidate to one canonical voice loop entry point

**Description:** As a developer, I want the voice loop duplication between root-level and
`rex/` package resolved so that there is one documented, authoritative startup path.

**Acceptance Criteria:**
- [ ] `CLAUDE.md` documents which file is the canonical voice loop entry point and why
  the second exists (or it is removed if unused)
- [ ] `rex_loop.py` (root) explicitly imports from the canonical module and does not
  duplicate business logic
- [ ] Both `voice_loop.py` files have a header comment explaining their relationship
- [ ] `pytest -q` exits 0
- [ ] Typecheck passes

---

# PHASE G — Dependency Alignment (DEP-001)

### US-193: Define and document the canonical runtime matrix

**Description:** As a developer, I want one authoritative version matrix for Python,
PyTorch, and CUDA targets so that requirements files, pyproject.toml, the Dockerfile,
and the validation script all agree.

**Acceptance Criteria:**
- [ ] `docs/DEPENDENCIES.md` (new or updated) documents the canonical matrix:
  Python 3.10–3.13, torch 2.6.x–2.8.x (CPU), cu118 variant, cu124 variant
- [ ] `requirements-cpu.txt` does not pin a CUDA variant of torch
- [ ] `pyproject.toml` optional ML extras match the documented matrix
- [ ] `Dockerfile` torch version falls within the documented matrix
- [ ] `scripts/validate_deployment.py` (from US-184) validates against the documented matrix
- [ ] Typecheck passes

---

# PHASE H — Code Bug Fixes (from code review)

### US-194: Thread-safe TTS engine in rex_speak_api.py

**Description:** As an operator, I want the TTS engine to be protected by a threading lock
so that concurrent HTTP requests to `/speak` cannot cause race conditions or corrupt state.

**Acceptance Criteria:**
- [ ] `rex_speak_api.py` introduces `_tts_lock = threading.Lock()` at module level
- [ ] All access to `_TTS_ENGINE` (read and write) is wrapped in `with _tts_lock:`
- [ ] The lock is acquired before initialization check and released after synthesis completes
- [ ] A test in `tests/test_speak_api.py` (or new file) fires 5 concurrent `/speak` requests
  using `threading.Thread` and asserts all return 200 with non-empty audio
- [ ] Typecheck passes

---

### US-195: Fix _followup_injected race condition in assistant.py

**Description:** As a developer, I want the followup injection flag in `assistant.py` to
use an `asyncio.Lock` so that concurrent `generate_reply` calls cannot inject followup
context twice.

**Acceptance Criteria:**
- [ ] `Assistant` replaces `self._followup_injected: bool` with
  `self._followup_lock: asyncio.Lock` initialized in `__init__`
- [ ] The injection block is wrapped in `async with self._followup_lock:`
- [ ] A test simulates two concurrent `generate_reply` calls and asserts followup context
  is injected at most once across both calls
- [ ] `pytest -q tests/test_assistant.py` exits 0
- [ ] Typecheck passes

---

### US-196: Fix inconsistent temp file cleanup in voice_loop.py

**Description:** As a developer, I want all temporary audio files created during
speech processing to be cleaned up reliably so that orphaned files do not accumulate.

**Acceptance Criteria:**
- [ ] All `tempfile.NamedTemporaryFile` and manual temp-path usages in `rex/voice_loop.py`
  and root `voice_loop.py` are wrapped in `try/finally` blocks that call `os.unlink()`
- [ ] Cleanup catches and logs `OSError` / `PermissionError` instead of raising
- [ ] A test creates a voice processing cycle with a mock audio source and asserts the
  temp directory has no leftover `.wav` files after the call
- [ ] `pytest -q tests/test_voice_loop.py` exits 0
- [ ] Typecheck passes

---

### US-197: Process OpenAI tool_calls in LLM client

**Description:** As a developer, I want the OpenAI strategy to correctly handle
`tool_calls` in responses so that function-calling is not silently ignored.

**Acceptance Criteria:**
- [ ] `rex/llm_client.py` `OpenAIStrategy.generate()` checks `message.tool_calls` and,
  when present, serializes them into the response string as structured JSON tool call data
- [ ] A test mocks an OpenAI response with `tool_calls` and asserts the returned
  string contains the tool name and arguments
- [ ] No existing tests regress
- [ ] Typecheck passes

---

### US-198: Fix Ollama error message taxonomy

**Description:** As a developer, I want Ollama connection errors to be distinguished
from missing-model errors so that users see accurate troubleshooting messages.

**Acceptance Criteria:**
- [ ] `OllamaStrategy` catches `httpx.ConnectError` / `ConnectionRefusedError` and returns
  `"[Ollama: connection failed — is Ollama running?]"`
- [ ] `OllamaStrategy` catches 404 model-not-found responses and returns
  `"[Ollama: model '{model}' not found — run: ollama pull {model}]"`
- [ ] All other errors return `"[Ollama: unexpected error: {detail}]"`
- [ ] Tests cover all three error paths with mocked HTTP responses
- [ ] Typecheck passes

---

### US-199: Fix sentence splitting for abbreviations in TTS pipeline

**Description:** As a user, I want TTS sentence splitting to not break mid-sentence on
abbreviations like "Dr.", "Mr.", "e.g.", and "etc." so that speech sounds natural.

**Acceptance Criteria:**
- [ ] `rex/voice_loop.py` sentence splitter uses an abbreviation-aware approach
  (either an allowlist of common titles/abbreviations or NLTK `sent_tokenize` if available,
  with regex fallback)
- [ ] "Dr. Smith said the treatment works." is treated as a single sentence
- [ ] "She said it was great. He agreed." is correctly split into two sentences
- [ ] "e.g. this example." is treated as one sentence
- [ ] Existing TTS pipeline tests pass
- [ ] Typecheck passes

---

### US-200: Add request body size limit to rex_speak_api.py

**Description:** As an operator, I want the `/speak` endpoint to reject request bodies
over a configurable size limit so that the server cannot be exhausted by oversized payloads.

**Acceptance Criteria:**
- [ ] `rex_speak_api.py` reads `MAX_REQUEST_BYTES` from config (default: 64 KB)
- [ ] Requests where `Content-Length` exceeds `MAX_REQUEST_BYTES` are rejected with 413
  before the body is read
- [ ] Requests without `Content-Length` are read with a stream cap at `MAX_REQUEST_BYTES`
- [ ] A test sends a request body of 100 KB and asserts the response is 413
- [ ] Typecheck passes

---

### US-201: Fix suppressed JSON errors in identity.py

**Description:** As a developer, I want session file JSON parse errors to be logged as
warnings so that corrupted session files are detectable and diagnosable.

**Acceptance Criteria:**
- [ ] `rex/identity.py` `_load_session()` catches `json.JSONDecodeError` and logs a
  `logger.warning(f"Corrupted session file {path}, resetting: {e}")` before returning `{}`
- [ ] The function no longer silently swallows parse errors
- [ ] A test writes a malformed JSON session file and asserts: the function returns `{}`
  and a warning is logged
- [ ] Typecheck passes

---

# PHASE I — Conversation History Persistence

### US-202: Create SQLite conversation history schema and HistoryStore class

**Description:** As a developer, I want a `HistoryStore` backed by SQLite so that
conversation turns can be persisted and retrieved across restarts.

**Acceptance Criteria:**
- [ ] New file `rex/history_store.py` defines `HistoryStore` with:
  - `__init__(self, db_path: Path)` — creates/migrates the DB on first call
  - `save_turn(user_id: str, role: str, content: str, timestamp: datetime) -> None`
  - `load_history(user_id: str, limit: int = 50) -> list[dict]`
  - `prune(user_id: str, keep_days: int = 30) -> int` — returns rows deleted
- [ ] Schema uses a single `turns` table: `(id, user_id, role, content, timestamp)`
- [ ] DB is created at `data/history.db` by default, path is configurable
- [ ] `pytest -q tests/test_history_store.py` exits 0 (new test file covers CRUD + prune)
- [ ] Typecheck passes

---

### US-203: Wire HistoryStore into assistant.py

**Description:** As a user, I want conversation history to survive assistant restarts so
that Rex can reference previous exchanges without the session being lost.

**Acceptance Criteria:**
- [ ] `Assistant.__init__` instantiates `HistoryStore` if `config.persist_history` is True
  (default: True)
- [ ] `Assistant.generate_reply()` calls `history_store.save_turn()` for each user prompt
  and assistant response
- [ ] `Assistant.__init__` preloads the last 50 turns from `HistoryStore` into the in-memory
  history on startup
- [ ] `AppConfig` gains a `persist_history: bool = True` field
- [ ] `pytest -q tests/test_assistant.py` exits 0
- [ ] Typecheck passes

---

### US-204: Add history rotation/pruning scheduled task

**Description:** As an operator, I want history older than a configurable retention
window to be pruned automatically so that the database does not grow unbounded.

**Acceptance Criteria:**
- [ ] `AppConfig` gains `history_retention_days: int = 30`
- [ ] `rex/history_store.py` `prune()` is called by the scheduler (or a startup hook)
  once per day
- [ ] Pruning is idempotent — running twice produces the same result as running once
- [ ] A test asserts that turns older than retention window are deleted and recent turns
  are preserved
- [ ] Typecheck passes

---

# PHASE J — Integration Backends

### US-205: Add transport-layer interfaces for email, calendar, and SMS

**Description:** As a developer, I want explicit protocol/ABC interfaces for each
integration backend so that real and stub implementations are interchangeable.

**Acceptance Criteria:**
- [ ] `rex/integrations/email/backends/base.py` defines `EmailBackend` ABC with:
  `fetch_unread(limit: int) -> list[dict]` and `send(to, subject, body) -> None`
- [ ] `rex/integrations/calendar/backends/base.py` defines `CalendarBackend` ABC with:
  `get_upcoming(days: int) -> list[dict]` and `create_event(title, start, end) -> dict`
- [ ] `rex/integrations/messaging/backends/base.py` defines `SMSBackend` ABC with:
  `send(to, body) -> None` and `receive() -> list[dict]`
- [ ] Existing mock/stub implementations are refactored to implement these interfaces
- [ ] `pytest -q` exits 0 (no regressions)
- [ ] Typecheck passes

---

### US-206: Implement IMAP read backend

**Description:** As a user, I want Rex to fetch real email from an IMAP server so that
the "read my email" command returns live inbox contents.

**Acceptance Criteria:**
- [ ] New file `rex/integrations/email/backends/imap_smtp.py` defines `IMAPBackend`
  implementing `EmailBackend.fetch_unread()`
- [ ] Uses stdlib `imaplib.IMAP4_SSL` with configurable host, port, and SSL flag
- [ ] Connection timeout is enforced (default: 10 s)
- [ ] On auth failure, raises a descriptive `EmailAuthError` (not a raw exception)
- [ ] `tests/test_email_backend_imap_smtp.py` tests happy-path and auth-failure cases
  using `unittest.mock` on the socket layer (no live network calls)
- [ ] `pytest -q tests/test_email_backend_imap_smtp.py` exits 0
- [ ] Typecheck passes

---

### US-207: Implement SMTP send backend

**Description:** As a user, I want Rex to send real email via SMTP so that "send email"
commands deliver to the actual recipient.

**Acceptance Criteria:**
- [ ] `IMAPSMTPBackend` (same file as US-206) implements `EmailBackend.send()` using
  stdlib `smtplib.SMTP` with STARTTLS or `smtplib.SMTP_SSL`
- [ ] Credentials are loaded via `CredentialManager` using the account's `credential_ref`
- [ ] Sensitive data (password) is never logged
- [ ] Tests cover: successful send, auth failure, TLS failure, and timeout — all with mocks
- [ ] `pytest -q tests/test_email_backend_imap_smtp.py` exits 0
- [ ] Typecheck passes

---

### US-208: Add multi-account email config and routing

**Description:** As a user, I want to configure multiple email accounts and have Rex route
reads/sends to the correct account so that work and personal email are separate.

**Acceptance Criteria:**
- [ ] `AppConfig` (via `rex_config.json`) supports:
  `email.accounts[]` (list of account objects with `id`, `address`, `imap`, `smtp`,
  `credential_ref`) and `email.default_account_id`
- [ ] `EmailService` accepts optional `account_id` on `fetch_unread()` and `send()`;
  falls back to `default_account_id` when omitted
- [ ] Invalid `account_id` raises `ValueError` with an actionable message
- [ ] Backward-compatible when only a single legacy account config is present
- [ ] `pytest -q tests/test_email_multi_account.py` exits 0 (new test file)
- [ ] Typecheck passes

---

### US-209: Wire notification email channel to real send backend

**Description:** As a user, I want urgent and digest notifications to be delivered via
the real SMTP backend so that I receive email alerts.

**Acceptance Criteria:**
- [ ] `rex/notification.py` `_send_to_email()` replaces the `"Would send."` log with a
  real call to `EmailService.send()`
- [ ] Digest flush also dispatches through `EmailService`
- [ ] On send failure, notification is marked failed (not silently dropped) and logged
- [ ] `pytest -q tests/test_notification_email_delivery.py` exits 0 (new test file)
- [ ] Typecheck passes

---

### US-210: Implement ICS calendar read-only feed backend

**Description:** As a user, I want Rex to read events from an ICS file or URL feed so
that the "upcoming events" command returns real calendar data.

**Acceptance Criteria:**
- [ ] New file `rex/integrations/calendar/backends/ics_feed.py` defines `ICSFeedBackend`
  implementing `CalendarBackend.get_upcoming()`
- [ ] Accepts a local file path or HTTP URL as the feed source
- [ ] Normalizes event timezones to UTC internally
- [ ] Handles malformed VEVENT blocks gracefully (logs warning, skips entry)
- [ ] Uses stdlib only (no `icalendar` package) for base parsing; falls back to `icalendar`
  if installed
- [ ] `tests/test_calendar_ics_backend.py` uses fixture `.ics` files; no live HTTP
- [ ] `pytest -q tests/test_calendar_ics_backend.py` exits 0
- [ ] Typecheck passes

---

### US-211: Implement Twilio SMS send adapter

**Description:** As a user, I want Rex to send real SMS messages via Twilio so that the
"send SMS" command delivers to the recipient's phone.

**Acceptance Criteria:**
- [ ] New file `rex/integrations/messaging/backends/twilio_sms.py` defines `TwilioSMSBackend`
  implementing `SMSBackend.send()`
- [ ] Uses the `twilio` optional extra; imports are guarded with a helpful error if not installed
- [ ] Credentials (`account_sid`, `auth_token`, `from_number`) loaded via `CredentialManager`
- [ ] On 4xx response: raises `SMSSendError` with Twilio error code
- [ ] On network timeout: raises `SMSSendError` with timeout detail
- [ ] No secrets logged at any log level
- [ ] `pytest -q tests/test_twilio_sms_backend.py` exits 0 with mocked Twilio client
- [ ] Typecheck passes

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

---

# PHASE K — Active CI Failures (Run 23946480448)

> **Context:** CI run 23946480448 exposed four failing jobs: Type Check (mypy), Python 3.11
> Tests & Coverage, Pre-commit Hook Validation, and indirectly the logo asset gap.
> Stories US-221 through US-229 fix every failure and bring the repo to production-ready
> state. They must be completed before US-220 is re-verified.

---

### US-221: Fix mypy no-redef error in rex/wakeword/embedding.py

**Description:** As a developer, I want the duplicate `_torch` symbol definition in
`embedding.py` removed so that mypy passes with zero no-redef errors in that file.

**Acceptance Criteria:**
- [ ] `rex/wakeword/embedding.py` defines `_torch` exactly once — either inside the
  `try` block or via a single `TYPE_CHECKING` guard, not both
- [ ] The torch import still degrades gracefully when torch is not installed
  (no `ImportError` at module load time)
- [ ] `mypy rex/wakeword/embedding.py --ignore-missing-imports` exits 0
- [ ] `pytest -q tests/` exits 0 (no regressions)
- [ ] Typecheck passes

---

### US-222: Fix mypy no-any-return errors in custom_voices.py and audio/smart_speaker_output.py

**Description:** As a developer, I want functions with declared return types of `float`
or `str` to return explicitly typed values so that mypy's `no-any-return` rule is satisfied.

**Acceptance Criteria:**
- [ ] `rex/custom_voices.py` line ~52: return value from `soundfile.info().duration`
  is cast to `float` (e.g. `return float(info.duration)`)
- [ ] `rex/audio/smart_speaker_output.py` line ~41: return value declared `float` is cast
  with `float(...)` before returning
- [ ] `rex/audio/smart_speaker_output.py` line ~52: return value declared `str` is cast
  with `str(...)` before returning
- [ ] `mypy rex/custom_voices.py rex/audio/smart_speaker_output.py --ignore-missing-imports`
  exits 0
- [ ] `pytest -q` exits 0
- [ ] Typecheck passes

---

### US-223: Remove stale type:ignore comments in compat, audio, shopping_pwa, and voice_loop

**Description:** As a developer, I want all `# type: ignore` comments that mypy now flags
as `[unused-ignore]` removed so that the annotation layer is clean and maintainable.

**Acceptance Criteria:**
- [ ] `rex/compat/transformers_shims.py` line ~76: stale `# type: ignore` removed (or
  replaced with a scoped `# type: ignore[attr-defined]` if the attribute access genuinely
  needs suppression — document why with an inline comment)
- [ ] `rex/audio/smart_speaker_output.py` line ~87: stale `# type: ignore` removed
- [ ] `rex/shopping_pwa.py` lines ~337, 361, 369, 378, 385, 399, 413: all seven stale
  `# type: ignore` comments removed
- [ ] `rex/voice_loop.py` line ~209: stale `# type: ignore` removed
- [ ] `mypy rex/compat/transformers_shims.py rex/audio/smart_speaker_output.py rex/shopping_pwa.py rex/voice_loop.py --ignore-missing-imports`
  exits 0 with zero `[unused-ignore]` errors
- [ ] `pytest -q` exits 0
- [ ] Typecheck passes

---

### US-224: Fix mypy return-value and call-arg errors in shopping_pwa.py and twilio_handler.py

**Description:** As a developer, I want Flask route handlers to return `flask.wrappers.Response`
(not `werkzeug.wrappers.response.Response`) and the `Assistant` constructor to be called
with its supported arguments so that mypy's `[return-value]` and `[call-arg]` rules pass.

**Acceptance Criteria:**
- [ ] `rex/shopping_pwa.py` line ~339: the route handler return type is corrected — either
  import and return `flask.Response(...)` directly, or cast via
  `typing.cast(flask.wrappers.Response, ...)`, with a comment explaining the approach
- [ ] `rex/telephony/twilio_handler.py` line ~88: the function declared to return `bool`
  returns `bool(...)` explicitly so `[no-any-return]` is resolved
- [ ] `rex/telephony/twilio_handler.py` line ~399: the `Assistant(config=...)` call is
  corrected to match `Assistant.__init__`'s actual signature (remove or rename the
  unsupported `config` keyword argument; check `rex/assistant.py` for the real parameter name)
- [ ] `mypy rex/shopping_pwa.py rex/telephony/twilio_handler.py --ignore-missing-imports`
  exits 0
- [ ] `pytest -q` exits 0
- [ ] Typecheck passes

---

### US-225: Fix psutil ModuleNotFoundError blocking CI test collection

**Description:** As a developer, I want `rex/tools/windows_diagnostics.py` to import
`psutil` conditionally so that test collection does not fail on Linux CI runners where
psutil is absent, and I want psutil added to dev requirements so Windows diagnostics
tests can run locally.

**Acceptance Criteria:**
- [ ] `rex/tools/windows_diagnostics.py` wraps `import psutil` in a `try/except ImportError`
  block; when psutil is unavailable, module-level functions return a descriptive
  `"psutil not installed"` fallback dict and a `logger.warning` is emitted
- [ ] `tests/test_windows_diagnostics.py` adds a `pytest.importorskip("psutil")` guard
  at the top of the file (or individual test functions) so the test is skipped gracefully
  on environments without psutil
- [ ] `requirements-dev.txt` (or `requirements-cpu.txt`) adds `psutil>=5.9` so psutil is
  available in CI and local dev installs
- [ ] `pytest -q tests/test_windows_diagnostics.py` exits 0 (skipped or passing — not erroring)
  on a Linux runner without psutil installed
- [ ] `pytest -q` exits 0 on a full run
- [ ] Typecheck passes

---

### US-226: Suppress pre-commit secret-detection false positives in docs and test fixtures

**Description:** As a developer, I want lines in documentation and test helpers that
detect-secrets flags as secrets to carry inline `# pragma: allowlist secret` (Python)
or `pragma: allowlist secret` (Markdown) suppression markers so that the pre-commit
hook exits clean on legitimate test fixtures and security-audit documentation.

**Acceptance Criteria:**
- [ ] `tests/helpers/fake_smtp.py` line ~15: the flagged line has `  # pragma: allowlist secret`
  appended (confirming it is a test credential, not a real secret)
- [ ] `tests/test_email_backend_imap_smtp.py` line ~602: same treatment
- [ ] `docs/ARCHITECTURE.md` line ~448: same treatment using Markdown comment syntax
  `<!-- pragma: allowlist secret -->` on the flagged line or the line above
- [ ] `docs/security/SECURITY_AUDIT_2026-01-08.md` lines ~40–41: both flagged lines
  suppressed with inline markers
- [ ] `pre-commit run detect-secrets --all-files` exits 0 after changes
- [ ] No real credentials exist at these locations (verify content is placeholder/example only)
- [ ] Typecheck passes

---

### US-227: Add AskRex brand logo assets to the repository

**Description:** As a developer, I want all official AskRex brand logo variants stored
under `assets/brand/` so that they are available to the GUI, README, shopping PWA, and
any other surface that needs them.

**Acceptance Criteria:**
- [ ] Directory `assets/brand/` is created
- [ ] The following logo variants are present as PNG files at `@2x` resolution (minimum
  512 px on the longest axis) plus one SVG where a vector source is available:
  - `icon-square.png` — T-Rex skull icon on dark rounded-square background
  - `icon-circle.png` — T-Rex skull icon on circular background
  - `icon-r.png` — R lettermark with embedded T-Rex silhouette
  - `wordmark-dark.png` — "AskRex" text on dark pill background
  - `wordmark-light.png` — "AskRex" text on transparent/white background
  - `wordmark-reverse.png` — "AskRex" text on white pill background, dark border
  - `primary-horizontal.png` — full horizontal lockup (icon + wordmark side-by-side)
  - `stacked.png` — full stacked lockup (icon above wordmark)
  - `favicon.ico` — 16/32/48 px multi-size favicon derived from `icon-square.png`
- [ ] `assets/brand/README.md` documents each variant, intended use, and minimum clear-space
  guidelines
- [ ] `assets/logo.svg` is updated or superseded by the canonical vector source if a higher-
  fidelity version is available
- [ ] Typecheck passes (no Python changes required; this is an asset story)

---

### US-228: Update README.md to use official AskRex brand logo

**Description:** As a user visiting the repository, I want to see the official AskRex
logo in the README so that the project presents a professional, branded identity.

**Acceptance Criteria:**
- [ ] `README.md` opens with an `<img>` tag (or Markdown image) referencing
  `assets/brand/primary-horizontal.png` (or `stacked.png`) at a display width of 400 px
- [ ] The image has a descriptive `alt` attribute: `"AskRex — local-first AI assistant"`
- [ ] The existing placeholder logo reference (if any) is removed
- [ ] README renders correctly on GitHub (verify via `gh browse` or manual inspection)
- [ ] Typecheck passes

---

### US-229: Update Electron GUI and shopping PWA to use official brand assets

**Description:** As a user running the desktop GUI or accessing the shopping PWA, I want
the official AskRex icon and wordmark to appear in the application chrome so that the
branded experience is consistent across all surfaces.

**Acceptance Criteria:**
- [ ] Electron app `package.json` (under `gui/`) sets `"icon"` to the path of
  `assets/brand/icon-square.png` (or `.ico` on Windows)
- [ ] Electron `BrowserWindow` creation passes the brand icon path for the taskbar/dock icon
- [ ] Shopping PWA HTML template (`rex/shopping_pwa/` or equivalent template file) includes
  the AskRex `icon-square.png` as `<link rel="icon">` and `<link rel="apple-touch-icon">`
- [ ] Shopping PWA `<title>` tag reads `"AskRex — Shopping"` (or similar brand-consistent name)
- [ ] Shopping PWA renders the `wordmark-dark.png` or `wordmark-light.png` in its header
- [ ] `pytest -q` exits 0 (no Python regressions)
- [ ] Typecheck passes

---

**Story ordering note (Phase K additions):**
- US-221 through US-224 are independent of each other and may run in parallel; all must
  complete before US-220 is re-verified
- US-225 (psutil) has no dependencies
- US-226 (secrets) has no dependencies
- US-227 must complete before US-228 and US-229
- US-228 and US-229 are independent of each other but both depend on US-227
- US-220 must remain the final verification story and should be re-run after all Phase K
  stories are complete
