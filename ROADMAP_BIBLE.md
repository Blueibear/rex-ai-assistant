# Rex AI Assistant Roadmap Bible
Last updated: 2026-02-21

This is the canonical execution plan for completing Rex AI Assistant. It is ordered by dependency and risk, and designed to be executed as prompt cycles:

- Cycle A: Claude Code implements a bounded slice on a feature branch, updates docs (including CLAUDE.md), and runs validation.
- Cycle B: Codex verifies claims, re-runs validation, fixes issues if needed, and produces a verification report.

## Current state
### Completed major milestones
- Email: multi-account foundation plus real IMAP and SMTP backend plus notification email channel wired (opt-in, offline-safe).
- Calendar: ICS read-only backend plus identity fallback CLI (whoami, identify) plus SSRF hardening for ICS URL sources.
- Messaging: Twilio send backend plus dashboard notification store plus notification API endpoints.
- Messaging inbound: Twilio webhook blueprint plus signature verification (stdlib) plus inbound store plus deterministic receive merge ordering.
- Voice identity: speaker recognition scaffolding plus fallback flow plus optional extras guarded behind import checks.

### Baseline rules
- Ruff and Black are non-negotiable. New or touched files must pass ruff check and black --check.
- CI enforces Conventional Commits for commit messages and PR title.
- Tests must keep the repo working tree clean (integrity tests fail otherwise).
- Pipenv and Dependabot safety: do not add heavy ML or CUDA deps to Pipfile. Keep pipenv lock --clear viable on clean Linux.

## Phase 4: Productionize inbound SMS and notifications
Goal: make inbound SMS and notifications usable end-to-end in a real deployment, without weakening security.

### Cycle 4.1 (Claude) Register inbound webhook in the running server
Scope
- Register the inbound webhook blueprint in the running Flask server (dashboard app or a dedicated webhook server).
- Startup wiring must:
  - Load config
  - Resolve Twilio auth token via CredentialManager
  - Initialize inbound store
  - Register POST /webhooks/twilio/sms
- Add rate limiting for the webhook route (reuse existing limiter patterns if present).
- Update doctor.py to validate webhook config safely.

Acceptance
- Endpoint is reachable and protected by Twilio signature verification (and rate limit).
- All tests are offline and deterministic (mock Flask requests, no network).

Validation
- python -m pip install -e ".[dev]"
- python -m ruff check <changed files>
- python -m black --check <changed files>
- pytest -q
- python scripts/doctor.py
- python scripts/security_audit.py

### Cycle 4.2 (Codex) Verify server wiring and security posture
- Verify blueprint registration location, proxy URL derivation, and rate limiting.
- Deliver VERIFICATION_REPORT_PHASE4_WEBHOOK_WIRING.md.

### Cycle 4.3 (Claude) User association for inbound SMS (route to user profiles)
Problem
- Inbound SMS can route to an account by To number, but does not reliably map messages to a user.

Scope
- Add user-to-phone mapping to config, for example:
  - identity.users[].phone_numbers[]
  - or messaging.accounts[].owner_user_id
- Persist user_id on inbound messages at ingest time.
- Update SMSService.receive() and CLI rex msg receive to support filtering by user.

### Cycle 4.4 (Codex) Verify user routing
- Add edge-case tests and deliver VERIFICATION_REPORT_PHASE4_USER_ROUTING.md.

### Cycle 4.5 (Claude) Minimal notification inbox UI
Scope
- Add a dashboard UI page that lists notifications from DashboardStore.
- Support mark read and mark all read.
- No realtime push yet. Polling is fine.

### Cycle 4.6 (Codex) Verify UI/API/persistence
- Deliver VERIFICATION_REPORT_PHASE4_NOTIFICATION_UI.md.

## Phase 5: Windows computer control (desktop and laptop)
Goal: Rex can safely take actions on your Windows machines (commands, files, status), with audit logs and policy gating.

### Cycle 5.1 (Claude) Windows endpoint contract plus config plus CLI
Scope
- Define config:
  - computers[]: {id, label, base_url, auth_token_ref, allowlists, enabled}
- Add CLI:
  - rex pc list
  - rex pc status --id <id>
  - rex pc run --id <id> -- <command> (must go through allowlist and policy approval)
- Implement a client in Rex that calls the agent API.
- Tests: offline, use a fake server.

### Cycle 5.2 (Codex) Verify safety boundaries
- Deliver VERIFICATION_REPORT_PHASE5_WINDOWS_CLIENT.md.

### Cycle 5.3 (Claude) Reference Windows agent implementation
Scope
- Minimal agent server with:
  - /health, /status, /run
- Secure defaults:
  - localhost binding by default
  - token auth
  - allowlist enforcement
  - rate limit
- Windows install docs (Scheduled Task or service).

### Cycle 5.4 (Codex) Verify agent and docs
- Deliver VERIFICATION_REPORT_PHASE5_WINDOWS_AGENT.md.

## Phase 6: WordPress plus WooCommerce integration
Goal: monitor and take safe actions in WordPress and WooCommerce with explicit approvals for write actions.

### Cycle 6.1 (Claude) Read-only monitoring
Scope
- Config:
  - wordpress.sites[]: {id, base_url, auth_method, credential_ref, enabled}
  - woocommerce.sites[]: {id, base_url, consumer_key_ref, consumer_secret_ref, enabled}
- CLI:
  - rex wc orders list --site <id>
  - rex wc products list --site <id> [--low-stock]
- Tests: mocked HTTP.

### Cycle 6.2 (Codex) Verify security and correctness
- Deliver VERIFICATION_REPORT_PHASE6_WP_WC_READONLY.md.

### Cycle 6.3 (Claude) Write actions with policy approval
Scope
- Add write commands gated by approvals (order status, coupons).
- Add tests proving gating.

### Cycle 6.4 (Codex) Verify policy gating
- Deliver VERIFICATION_REPORT_PHASE6_WP_WC_WRITE.md.

## Phase 7: Voice identity MVP (beyond scaffolding)
### Cycle 7.1 (Claude) Enrollment and calibration
- Enrollment CLI and embedding storage per user.
- Optional deps only under voice-id extras.

### Cycle 7.2 (Codex) Verify failure safety
- Deliver VERIFICATION_REPORT_PHASE7_VOICE_ID_MVP.md.

## Phase 8: Realtime notifications and remaining hardening
### Cycle 8.1 (Claude) Realtime push (SSE first)
### Cycle 8.2 (Codex) Verify SSE auth and stability
### Cycle 8.3 (Claude) Optional follow-ups
- Calendar RRULE and TZID upgrades, CalDAV adapter.
- Home Assistant TTS channel.

## Reference
- BACKLOG.md is the original planning snapshot that seeded this bible.
