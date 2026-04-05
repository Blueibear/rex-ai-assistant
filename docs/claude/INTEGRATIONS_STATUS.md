
---

## `docs/claude/INTEGRATIONS_STATUS.md`

```md
# Claude Reference: Integrations Status

This file is the single source of truth for integration readiness.
Each integration is classified as **REAL**, **PARTIAL**, **STUB**, or **NOT STARTED**.

- **REAL** — backend is complete, tested, and usable in production with credentials.
- **PARTIAL** — backend exists but has meaningful gaps (read-only, credential-gated fallback, or missing hardening).
- **STUB** — scaffolding exists; no real external service connection.
- **NOT STARTED** — feature is in the roadmap only; no code exists.

---

## Integration Status Snapshot

### Email
**Status: PARTIAL**
Evidence: Real IMAP4-SSL read and SMTP send backend exists (`rex/integrations/email/backends/imap_smtp.py`). Multi-account routing exists. Falls back to offline stub when IMAP/SMTP credentials are absent. CalDAV/Google OAuth not implemented.

### Calendar
**Status: PARTIAL**
Evidence: ICS read-only backend exists (`rex/integrations/calendar/backends/ics_feed.py`). Supports local `.ics` files and HTTPS ICS sources. No calendar write support. CalDAV/Google OAuth not implemented.

### SMS / Messaging
**Status: PARTIAL**
Evidence: Real SMS delivery and inbound webhook receiver via Twilio exists (`rex/integrations/messaging/backends/twilio_sms.py`). Multi-account routing and inbound message handling included. Falls back to offline stub when Twilio credentials are absent.

### Notifications
**Status: REAL**
Evidence: Priority routing, digest logic, quiet hours, and auto-escalation are active. Dashboard channel persists to local SQLite store with real API endpoints and SSE push (`rex/notifications/`, `rex/dashboard_store.py`). Email channel uses real SMTP when configured.

### Voice Identity / Speaker Recognition
**Status: PARTIAL**
Evidence: Embeddings store and enrollment commands exist (`rex/voice_identity/`). Calibration and recognition scaffolding present. Still alpha-only; optional dependency model used.

### Windows Computer Control
**Status: PARTIAL**
Evidence: Windows agent server and client foundation exist (`rex/computers/`). Approval and allowlist model present. Boot-persistence and service-wrapper hardening are roadmap items.

### Home Assistant TTS
**Status: PARTIAL**
Evidence: Optional notification channel exists (`rex/ha_bridge.py`). Disabled by default. Auth and SSRF hardening are required for production use.

### WordPress / WooCommerce
**Status: PARTIAL**
Evidence: Read-only health check via WP REST API (`rex wp health`). Orders and products list via WC REST API v3 (`rex wc orders list`, `rex wc products list`). Write actions deferred to future cycle.

### Web Search
**Status: PARTIAL**
Evidence: Backends for SerpAPI, Brave, Google CSE, and DuckDuckGo exist (`rex/search/`). Requires API credentials; no credential = no results.

### OpenClaw Gateway
**Status: REAL**
Evidence: HTTP integration complete (Phase 8). All calls use `rex/openclaw/http_client.py` with retries and auth. Feature flags in `config/rex_config.json` under `openclaw` control voice-backend and tool-routing paths.

### Per-User Memory / Conversation History
**Status: PARTIAL**
Evidence: Per-user memory profiles exist (`Memory/`). Conversation history persistence is in progress and remains alpha-only.

### Autonomous Workflows / Planner
**Status: STUB**
Evidence: Workflow runner scaffolding exists (`rex/workflow_runner.py`, `rex/autonomy/`). Still alpha-only; roadmap item for future cycle.

### Identity (Session-Scoped Fallback)
**Status: PARTIAL**
Evidence: `rex identify` and `rex whoami` commands work for session-scoped identity. Full voice/speaker recognition is PARTIAL (see Voice Identity above).

---

## Known Caution Areas

- Autonomous tool execution and scheduler-triggered workflow claims are STUB level.
- Docker deployment guidance may not reflect latest Dockerfile state.
- Windows quickstart commands should be verified on target Python 3.11 environment.

## When to Include This File in a Task Packet

Include this file when the task touches:
- integration docs
- roadmap sequencing
- feature-readiness wording
- capability claims
- implementation status labels
```
