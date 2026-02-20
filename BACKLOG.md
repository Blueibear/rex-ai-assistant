# Backend Integration Backlog (Implementation-Ready)

## Implementation Plan

### Phase 1 (PR 1): Real Email + Notification Email Channel + Multi-account foundation
- Replace stub email read/send with IMAP4 over SSL + SMTP (STARTTLS/SSL) backend behind adapters.
- Add multi-account email config, active-account selection, and routing defaults.
- Wire notification email channel to real email send path while preserving priority/digest behavior.
- Keep tests fully local using IMAP/SMTP fakes and fixture transcripts (no live credentials).

### Phase 2 (PR 2): Calendar real backend + SMS real Twilio send path
- Replace calendar stub with ICS read-only feed backend first (safe, testable, low OAuth complexity).
- Complete Twilio SMS send path with strict failure handling and deterministic mock transport tests.
- Integrate calendar event polling into existing scheduler/event bus contract.

### Phase 3 (PR 3): Voice recognition architecture + optional heavy dependencies
- Add speaker recognition architecture + enrollment/verification scaffolding.
- Implement fallback identity flow when recognition confidence is below threshold.
- Keep speaker-recognition heavy ML dependencies behind optional extras only.

---

## A) Repo Mapping (Current state)

## Integration stubs currently present
- **Email**: `EmailService` is explicitly stub/mock based (`data/mock_emails.json`), and does not implement IMAP/SMTP send/read backends yet. (`rex/email_service.py`)
- **Calendar**: `CalendarService` persists mock events and documents real API as future work; scheduler consumes this mock source. (`rex/calendar_service.py`, `rex/integrations.py`)
- **SMS / messaging**: `SMSService` stores sent/received messages in `data/mock_sms.json`; Twilio is referenced but not used for real delivery. (`rex/messaging_service.py`)
- **Notifications**: routing/priority/digest/escalation logic is real, but channel delivery is mostly placeholder; email logs “Would send.” (`rex/notification.py`)

## Existing CLI commands and contracts
- **Email CLI**: `rex email unread [--limit N] [-v]` uses `EmailService.fetch_unread()`.
- **Calendar CLI**: `rex calendar upcoming [--days N] [--conflicts] [-v]` uses `CalendarService.get_upcoming_events()`.
- **Messaging CLI**: `rex msg send --channel sms --to ... --body ...` and `rex msg receive --channel sms` use `SMSService`.
- **Notifications CLI**: `rex notify send/list-digests/flush-digests/ack` uses `NotificationRequest` + `Notifier`.
- **Contracts/Interfaces**:
  - `MessagingService` ABC + `Message` pydantic model in `rex/messaging_service.py`.
  - Notification types in `rex/notification.py` and `rex/contracts/core.py`.
  - Scheduler/event contracts via `ScheduledJob`, `Event`, and event types `email.unread` / `calendar.update` in `rex/integrations.py`.

## Configuration and credential loading
- Runtime config is loaded from `config/rex_config.json` schema pattern (`rex/config.py`, `config/rex_config.example.json`).
- Secrets are loaded via credential manager env/config mapping (`rex/credentials.py`) from env and optional `config/credentials.json`.
- Current credential mapping has generic `email` token and no first-class multi-account IMAP/SMTP credential structure yet.

---

## B) Prioritized Backlog (implementation-ready)

## BL-001 — Introduce transport interfaces for email/calendar/sms
- **Scope**: Add explicit backend interfaces and adapters so services stop embedding stub storage logic directly.
- **Success criteria**:
  - `EmailService`, `CalendarService`, `SMSService` accept transport implementations (real or mock).
  - Existing CLI/API behavior remains backward compatible.
- **Files/modules to change**:
  - `rex/email_service.py`
  - `rex/calendar_service.py`
  - `rex/messaging_service.py`
- **New files**:
  - `rex/integrations/email/backends/base.py`
  - `rex/integrations/calendar/backends/base.py`
  - `rex/integrations/messaging/backends/base.py`
- **Tests**:
  - Add interface conformance tests for each transport.
- **Docs**:
  - `docs/email.md`, `docs/calendar.md`, `docs/notifications.md`
- **Validation commands**:
  - `pytest -q tests/test_email_service.py tests/test_calendar_service.py tests/test_messaging_service.py`
- **Dependency impact**:
  - None; stdlib/protocol abstraction only.

## BL-002 — Implement real email backend (IMAP read + SMTP send)
- **Scope**: Implement minimal production-safe email backend using stdlib `imaplib`, `smtplib`, and `ssl`.
- **Success criteria**:
  - Can fetch unread messages from IMAP inbox.
  - Can send email via SMTP with STARTTLS or SSL.
  - Timeouts, retry policy, and non-secret logging are enforced.
- **Files/modules to change**:
  - `rex/email_service.py`
  - `rex/retry.py` (reuse/extend for transport retries if needed)
  - `rex/credentials.py` (credential shape helpers)
- **New files**:
  - `rex/integrations/email/backends/imap_smtp.py`
  - `tests/test_email_backend_imap_smtp.py`
  - `tests/fixtures/email/imap_transcript/*.json`
- **Tests**:
  - Mock socket/imap/smtp interactions; zero real network.
  - Negative tests: auth failure, TLS failure, malformed message, timeout.
- **Docs**:
  - `docs/email.md`, `docs/credentials.md`, `CONFIGURATION.md`
- **Validation commands**:
  - `pytest -q tests/test_email_backend_imap_smtp.py tests/test_email_service.py`
- **Dependency impact**:
  - None; stdlib only (Dependabot/pipenv lock unaffected).

## BL-003 — Multi-account email config + account routing
- **Scope**: Allow multiple email accounts per user with active account selection and routing policy.
- **Success criteria**:
  - Config supports `email.accounts[]`, per-account auth/server settings, and `default_account_id`.
  - Read/send APIs accept optional `account_id`; default routing works.
  - Invalid account selection returns actionable errors.
- **Files/modules to change**:
  - `config/rex_config.schema.json`
  - `config/rex_config.example.json`
  - `rex/config.py`
  - `rex/email_service.py`
  - `rex/cli.py`
- **New files**:
  - `tests/test_email_multi_account.py`
- **Tests**:
  - Selection precedence tests (explicit account > user default > global default).
  - Backward compatibility when single-account legacy config is present.
- **Docs**:
  - `docs/email.md`, `CLAUDE.md`, `CONFIGURATION.md`
- **Validation commands**:
  - `pytest -q tests/test_email_multi_account.py tests/test_cli_scheduler_email_calendar.py`
- **Dependency impact**:
  - None.

## BL-004 — Notification email channel to real send backend
- **Scope**: Replace placeholder `_send_to_email` with actual `EmailService.send(...)`, preserving current priority/digest flow.
- **Success criteria**:
  - Urgent/normal/digest notifications can dispatch via configured email account.
  - Retry/idempotency semantics remain unchanged.
- **Files/modules to change**:
  - `rex/notification.py`
  - `rex/email_service.py`
  - `rex/integrations.py`
- **New files**:
  - `tests/test_notification_email_delivery.py`
- **Tests**:
  - Channel dispatch tests with fake email backend.
  - Digest flush to email tests.
- **Docs**:
  - `docs/notifications.md`, `docs/email.md`
- **Validation commands**:
  - `pytest -q tests/test_notification.py tests/test_notification_email_delivery.py tests/test_cli_messaging_notification.py`
- **Dependency impact**:
  - None.

## BL-005 — Add account-aware notification routing rules
- **Scope**: Notifications select destination email account by rule (e.g., metadata/account tag/user default).
- **Success criteria**:
  - Notification metadata may specify `email_account_id`.
  - Missing account falls back deterministically; logs route decision safely.
- **Files/modules to change**:
  - `rex/notification.py`
  - `rex/config.py`
- **New files**:
  - `tests/test_notification_routing_rules.py`
- **Tests**:
  - Routing decision matrix tests.
- **Docs**:
  - `docs/notifications.md`, `CONFIGURATION.md`
- **Validation commands**:
  - `pytest -q tests/test_notification_routing_rules.py`
- **Dependency impact**:
  - None.

## BL-006 — Calendar real backend (ICS read-only first)
- **Scope**: Implement read-only ICS feed backend as first production backend.
- **Success criteria**:
  - Imports events from local file path or URL feed.
  - Normalizes timezone and event IDs.
  - Publishes `calendar.update` events compatible with current notifier hooks.
- **Files/modules to change**:
  - `rex/calendar_service.py`
  - `rex/integrations.py`
  - `rex/config.py`
- **New files**:
  - `rex/integrations/calendar/backends/ics_feed.py`
  - `tests/test_calendar_ics_backend.py`
  - `tests/fixtures/calendar/*.ics`
- **Tests**:
  - Feed parse tests, recurrence edge cases (minimal supported subset), malformed ICS handling.
- **Docs**:
  - `docs/calendar.md`, `CONFIGURATION.md`
- **Validation commands**:
  - `pytest -q tests/test_calendar_ics_backend.py tests/test_calendar_service.py`
- **Dependency impact**:
  - Prefer stdlib parser strategy first; if adding `icalendar`, keep lightweight pin and validate pipenv lock compatibility.

## BL-007 — Twilio SMS send path completion
- **Scope**: Keep existing `SMSService` API, add Twilio transport adapter for real sends with robust fallbacks.
- **Success criteria**:
  - Production send path calls Twilio client when configured.
  - On failure, explicit error taxonomy and optional queue-to-local fallback behavior.
  - No secret leakage in logs.
- **Files/modules to change**:
  - `rex/messaging_service.py`
  - `rex/credentials.py`
- **New files**:
  - `rex/integrations/messaging/backends/twilio_sms.py`
  - `tests/test_twilio_sms_backend.py`
- **Tests**:
  - Twilio client mocked with `unittest.mock`.
  - 4xx/5xx/network timeout handling tests.
- **Docs**:
  - `docs/notifications.md`, `docs/credentials.md`
- **Validation commands**:
  - `pytest -q tests/test_twilio_sms_backend.py tests/test_messaging_service.py tests/test_cli_messaging_notification.py`
- **Dependency impact**:
  - Twilio package remains optional extra (`[project.optional-dependencies].sms`); no Pipfile default dependency.

## BL-008 — Add integration test harnesses with no real credentials
- **Scope**: Build deterministic mock fixtures and fake transports for IMAP/SMTP/Twilio/ICS.
- **Success criteria**:
  - CI/integration tests pass fully offline.
  - Each backend has happy-path + failure-path tests.
- **Files/modules to change**:
  - `conftest.py`
  - Existing integration tests under `tests/`
- **New files**:
  - `tests/helpers/fake_imap.py`
  - `tests/helpers/fake_smtp.py`
  - `tests/helpers/fake_twilio.py`
- **Tests**:
  - Add fixtures to isolate filesystem and network.
- **Docs**:
  - `CLAUDE.md`, `docs/developer_tools.md`
- **Validation commands**:
  - `pytest -q tests/test_email_backend_imap_smtp.py tests/test_twilio_sms_backend.py tests/test_calendar_ics_backend.py`
- **Dependency impact**:
  - None.

## BL-009 — Voice speaker recognition architecture + fallback identity flow
- **Scope**: Add design + scaffolding for enrollment, embedding store, recognition thresholds, and fallback prompt/verification.
- **Success criteria**:
  - New module boundaries defined and scaffolded.
  - Runtime supports “recognized speaker”, “uncertain speaker”, and “unknown speaker” branches.
  - Fallback identity flow ties into existing profile manager / active user logic.
- **Files/modules to change**:
  - `rex/voice_loop.py`
  - `rex/profile_manager.py`
  - `rex/config.py`
- **New files**:
  - `rex/voice_identity/types.py`
  - `rex/voice_identity/embeddings_store.py`
  - `rex/voice_identity/recognizer.py`
  - `rex/voice_identity/fallback_flow.py`
  - `docs/voice_identity.md`
  - `tests/test_voice_identity_fallback.py`
- **Tests**:
  - Enrollment and threshold decision unit tests with synthetic embeddings.
- **Docs**:
  - `CLAUDE.md`, `docs/README_STABILIZATION.md`
- **Validation commands**:
  - `pytest -q tests/test_voice_identity_fallback.py tests/test_profile_manager.py`
- **Dependency impact**:
  - No heavy deps in default install.

## BL-010 — Optional extras policy for speaker recognition
- **Scope**: Define optional extras (e.g., `voice-id`) for heavy speaker recognition libraries and guard imports.
- **Success criteria**:
  - Default `pipenv install` does not pull heavy ML/CUDA packages.
  - Runtime clearly reports missing optional deps and falls back cleanly.
- **Files/modules to change**:
  - `pyproject.toml`
  - `docs/DEPENDENCIES.md`
  - `CLAUDE.md`
- **New files**:
  - `tests/test_optional_voice_id_imports.py`
- **Tests**:
  - Import guard tests ensuring base install remains operational.
- **Docs**:
  - `README.md`, `INSTALL.md`
- **Validation commands**:
  - `pytest -q tests/test_optional_voice_id_imports.py`
  - `pipenv lock --clear`
- **Dependency impact**:
  - Heavy voice libs optional only; Pipfile/Pipfile.lock remain minimal and lockable.

## BL-011 — CLI additions for account management and backend diagnostics
- **Scope**: Add practical CLI workflows for integration setup and troubleshooting.
- **Success criteria**:
  - New commands for email account listing/selecting/testing.
  - Diagnostic commands return actionable failures without exposing secrets.
- **Files/modules to change**:
  - `rex/cli.py`
  - `scripts/doctor.py`
- **New files**:
  - `tests/test_cli_email_accounts.py`
- **Tests**:
  - CLI parser + behavior tests.
- **Docs**:
  - `CLAUDE.md`, `docs/email.md`, `docs/QUICK_REFERENCE.md`
- **Validation commands**:
  - `pytest -q tests/test_cli_email_accounts.py tests/test_cli.py tests/test_doctor.py`
- **Dependency impact**:
  - None.

## BL-012 — Docs/status alignment and implementation status enforcement
- **Scope**: Keep integration docs accurate and status-tagged across email/calendar/notifications/messaging.
- **Success criteria**:
  - Each integration doc has top-level status and backend matrix.
  - Commands/config snippets match actual implementation.
- **Files/modules to change**:
  - `docs/email.md`
  - `docs/calendar.md`
  - `docs/notifications.md`
  - `docs/credentials.md`
  - `CLAUDE.md`
- **New files**:
  - None required.
- **Tests**:
  - Optional doc lint checks if available.
- **Docs**:
  - N/A (this item is docs).
- **Validation commands**:
  - `pytest -q tests/test_repository_integrity.py`
- **Dependency impact**:
  - None.

---

## C) Backend choice recommendations (simplest viable first)

## Email backend choice
- **Recommended primary**: IMAP4 over SSL (`imaplib.IMAP4_SSL`) for read + SMTP (`smtplib.SMTP` + STARTTLS, or `SMTP_SSL`) for send.
- **Why**:
  - Standard protocol, provider-agnostic, no SDK lock-in.
  - Uses Python stdlib only, keeping Pipfile unchanged.
  - Easy to mock for local tests.

## Credential storage recommendation
- **Primary approach**: Keep secrets in env vars / `config/credentials.json` via existing `CredentialManager`, with account-scoped keys (e.g., `email:primary`, `email:work`) and optional encrypted-at-rest support delegated to OS secret manager later.
- **Fallback approach**: App-password-only `.env`/CI secret injection for non-interactive flows (CI/staging), plus explicit warning if plaintext file storage is used.
- **Windows practicality**: Prefer environment + optional keyring bridge in future; do not block phase 1 on keyring adoption.

## Multi-account config proposal
```json
{
  "email": {
    "default_account_id": "personal",
    "accounts": [
      {
        "id": "personal",
        "address": "you@example.com",
        "imap": {"host": "imap.example.com", "port": 993, "ssl": true},
        "smtp": {"host": "smtp.example.com", "port": 587, "starttls": true},
        "credential_ref": "email:personal"
      },
      {
        "id": "work",
        "address": "you@company.com",
        "imap": {"host": "imap.company.com", "port": 993, "ssl": true},
        "smtp": {"host": "smtp.company.com", "port": 587, "starttls": true},
        "credential_ref": "email:work"
      }
    ]
  }
}
```
Routing precedence:
1. explicit `account_id` argument,
2. per-notification metadata `email_account_id`,
3. user-level default,
4. global `default_account_id`.

## Calendar backend choice
- **Recommended first real backend**: **ICS read-only feed import**.
- **Justification**:
  - Lowest auth/security complexity (no OAuth token lifecycle in phase 1/2).
  - Works with local file fixtures and HTTP mocks for deterministic tests.
  - Fastest path to “real” data integration while preserving scheduler/event bus design.
- **Follow-on**: add CalDAV second, Google OAuth third.

## Notifications delivery choice
- Implement **email delivery** first using BL-002 backend and current Notifier priority/digest mechanics unchanged.

## Voice recognition architecture recommendation
- **Enrollment**: capture N samples per user and compute normalized speaker embeddings.
- **Storage**: versioned embedding store per user profile (`Memory/<user>/voice_embeddings.json` or similar) with metadata (model version, sample count, updated_at).
- **Recognition**:
  - compute embedding for incoming utterance,
  - cosine similarity against enrolled users,
  - apply configurable thresholds:
    - `accept_threshold` (recognized),
    - `review_threshold` (uncertain).
- **Fallback identity flow**:
  - If below `review_threshold`: ask identity confirmation (voice/text PIN or quick prompt),
  - temporary session identity tag until strong signal appears,
  - audit events for recognition/fallback decisions.
- **Dependency policy**:
  - heavy libs (e.g., `speechbrain`, `resemblyzer`, torch-audio stacks) behind optional extras only; guarded imports + graceful no-op fallback.

---

## D) PR slicing strategy
- **PR 1 (Phase 1)**: BL-001, BL-002, BL-003, BL-004, BL-005 (+ docs/tests).
- **PR 2 (Phase 2)**: BL-006, BL-007, BL-008 (+ docs/tests).
- **PR 3 (Phase 3)**: BL-009, BL-010, BL-011, BL-012 (+ docs/tests).

Reasoning: keeps high-risk integration concerns separated by protocol domain (email first, then calendar/SMS, then voice optionality).

---

## Concrete validation commands by phase

## Phase 1
```bash
pytest -q tests/test_email_service.py tests/test_email_backend_imap_smtp.py tests/test_email_multi_account.py tests/test_notification.py tests/test_notification_email_delivery.py tests/test_cli_messaging_notification.py
```

## Phase 2
```bash
pytest -q tests/test_calendar_service.py tests/test_calendar_ics_backend.py tests/test_messaging_service.py tests/test_twilio_sms_backend.py tests/test_cli_scheduler_email_calendar.py
```

## Phase 3
```bash
pytest -q tests/test_voice_identity_fallback.py tests/test_optional_voice_id_imports.py tests/test_cli_email_accounts.py tests/test_profile_manager.py
pipenv lock --clear
```

---

## Top 10 backlog items (priority order)
1. BL-002 Real IMAP/SMTP email backend
2. BL-003 Multi-account config + routing
3. BL-004 Notification email delivery via real backend
4. BL-001 Transport interfaces for email/calendar/sms
5. BL-006 Calendar ICS real backend
6. BL-007 Twilio SMS real send adapter
7. BL-008 Offline integration test harnesses
8. BL-005 Account-aware notification routing
9. BL-009 Voice recognition architecture + fallback identity flow
10. BL-010 Optional extras policy for voice recognition heavy deps

## Recommended Phase 1 scope
- Implement BL-001 through BL-005 only.
- Do not add non-stdlib dependencies in Phase 1.
- Keep existing CLI commands working while adding account-aware options incrementally.

## Exact commands James should run to validate Phase 1
```bash
pytest -q tests/test_email_service.py tests/test_email_backend_imap_smtp.py tests/test_email_multi_account.py tests/test_notification.py tests/test_notification_email_delivery.py tests/test_cli_messaging_notification.py
python -m rex.cli email unread --limit 3
python -m rex.cli notify send --priority normal --title "phase1-smoke" --body "email channel smoke" --channels email
```
