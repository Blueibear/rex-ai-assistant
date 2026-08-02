# AskRex Mobile API Gateway Implementation Plan

Tracking issue: #323  
Planning date: 2026-07-14  
Required base: current `origin/master`, including `683b765ad6f4cc79771197f7390982487b0ac6c8`

## Delivery strategy

Cross-repository reconnaissance and contract decisions are complete in the planning PR. Implementation uses the minimum two Fable sessions that preserve most token savings while keeping security-critical changes reviewable.

| Session | Backend work | Mobile work |
|---|---|---|
| 1 | API process, config, middleware, status/capabilities, scaffolds, users, access/refresh auth, per-device sessions, revocation | No mobile merge required |
| 2 | HTTP/SSE/WebSocket chat, shared idempotency, voice, TTS, final capabilities/docs | Align shared auth client plus draft PR #3 and draft PR #5 |

Each backend session opens one focused PR against `master`. Do not merge without explicit authorization. Mobile PRs remain draft until matching backend integration is exercised.

## Rules for both sessions

- Read `CLAUDE.md` and every document under `docs/mobile/` before editing.
- Verify folder, worktree, branch, base SHA, remotes, and status.
- Never stage, restore, overwrite, or commit `.claude/ralph-loop.local.md`.
- Reuse canonical users, identity, permissions, Assistant, memory, policy, approvals, STT, and TTS.
- Validate identity before private access and pass it explicitly to Assistant.
- Never trust client identity, role, permissions, risk, approval, or biometric fields.
- Never present a scaffold, mock, request submission, or client biometric result as completed/verified.
- Use targeted reads and local logs; do not paste full sources or passing logs into chat.
- Run focused tests before full validation. Investigate failures and reproduce claimed base failures on an untouched exact-base worktree.
- Update `CLAUDE.md` when commands, config, dependencies, environment variables, integrations, or layout change.
- Verify tests leave the worktree clean.

---

# Session 1 — Foundation and Authentication

Recommended branch/worktree:

```text
fable/mobile-api-auth-323
C:\Users\james\rex-ai-test\rex-ai-mobile-api-auth
```

This session does not implement real chat, WebSocket, voice, TTS, Home Assistant, approvals, notifications, tasks, workflows, audit, or settings. Those routes return explicit 501 errors and false capabilities.

## S1-01: Mobile application factory

Create `rex/mobile_api/` with an injectable Flask app factory. Install request IDs, nested errors, privacy-safe logging, body limits, deny-by-default CORS, and rate limiting. Register only `/mobile/*` routes. Keep it separate from the Electron GUI/admin server. Imports must not start listeners or load heavy ML/audio dependencies.

## S1-02: Typed configuration

Add `MobileApiConfig` under `config.mobile_api` using the master-spec defaults. Validate host, port, TTLs, limits, origins, and API version. The original `.env` storage direction for `REX_JWT_SECRET` is superseded by the desktop credential vault policy; require adequate entropy and add no new flat config reads.

## S1-03: CLI commands

Implement:

```powershell
python -m rex mobile-api --host 127.0.0.1 --port 8765
python -m rex mobile-api --host 0.0.0.0 --port 8765
python -m rex mobile-user create --username james
```

CLI flags override config. Print bind/status data and an insecure-LAN warning without secrets. `mobile-user create` prompts twice with `getpass`, reuses `rex.auth.create_user`, creates the matching profile, and bootstraps canonical first-user permissions.

## S1-04: Canonical user-store migration

Reuse `data/users.db`. Add idempotent active-user support when absent and create `mobile_sessions` and `mobile_refresh_tokens` from the architecture document. Preserve existing users and hashes. Test fresh, legacy, and repeated migration.

## S1-05: Session and refresh lifecycle

Generate high-entropy opaque refresh tokens; store only SHA-256 hashes. Create per-device sessions. Rotate in one SQLite transaction. Exactly one concurrent refresh may succeed. Consumed-token reuse revokes the family/session. Implement idempotent current-session logout and authenticated logout-all. Inject clock/token generators for deterministic tests.

## S1-06: Access JWT and principal

Issue 15-minute JWTs containing `iss`, `aud`, `sub`, `sid`, `jti`, `iat`, `nbf`, and `exp`. Validate algorithm/signature/issuer/audience/time/required claims/session/user status. Validate canonical `sub` before private lookup. Resolve current permissions server-side and attach an immutable request principal. Never authorize from client or stale display claims.

## S1-07: Authentication routes

Implement exactly:

```text
POST /mobile/auth/login
POST /mobile/auth/refresh
POST /mobile/auth/logout
POST /mobile/auth/logout-all
GET  /mobile/auth/session
```

Match the master-spec shapes. Login creates a device session. Session returns live profile/role/permissions. Logout invalidates otherwise-unexpired access tokens through session checks. Apply route-specific limits and non-enumerating errors.

## S1-08: Status, capabilities, and scaffolds

Add minimal public `/mobile/status` and `/mobile/capabilities`. In Session 1, authentication is true and runtime features are false unless genuinely implemented. Register every requested unsupported route as an authenticated 501 `NOT_IMPLEMENTED` response. Return no fake data.

## S1-09: Documentation and smoke

Document secret generation, user creation, localhost/LAN startup, status, login, refresh, session, logout, Windows Firewall private-network setup, and troubleshooting. Run a real local status/auth/refresh/logout smoke. Do not claim iPhone validation unless performed.

## Session 1 tests

Cover:

- fresh/legacy/idempotent migrations;
- login success, invalid login, non-enumeration, malformed payload/content type/body limit;
- JWT signature, issuer, audience, expiry, nbf, required claims;
- missing/invalid/reserved IDs before private access;
- active/revoked/expired session and user status;
- refresh rotation, concurrency, reuse-family revocation;
- current/all logout and two-user isolation;
- live permission changes;
- request IDs, nested errors, rate limits, and log redaction;
- status/capabilities/scaffolds;
- CLI/config precedence and clean worktree.

## Session 1 validation

```powershell
python -m rex mobile-api --help
python -m rex mobile-user --help
pytest -q tests/mobile_api
pytest -q tests/test_us047_user_auth.py tests/test_us048_data_isolation.py tests/test_us052_permissions.py
ruff check .
black --check --diff rex/ tests/ bridge/ *.py
python -m compileall -q rex scripts
mypy rex --ignore-missing-imports
detect-secrets scan --baseline .secrets.baseline
pytest -q
python -m rex doctor
```

Also run current CI integration/security commands. Commit, push, open one PR referencing #323, wait for/fix CI, comment on #323, do not merge, and stop.

---

# Session 2 — Chat, WebSocket, Voice, TTS, and Client Alignment

Start only after Session 1 merges. Recommended backend branch/worktree:

```text
fable/mobile-api-runtime-323
C:\Users\james\rex-ai-test\rex-ai-mobile-api-runtime
```

Mobile worktree:

```text
C:\Users\james\rex-ai-test\askrex-mobile\AskRex
```

Update draft PR #3 and #5 rather than creating duplicate competing implementations unless reconciliation is impossible and documented.

## S2-01: Shared idempotency

Add `mobile_message_requests` and a SQLite service keyed by `(user_id, message_id)`. Hash normalized semantic request fields. Coordinate HTTP and WebSocket. Reserve before acknowledgement/execution, return stored exact duplicates, reject same ID/different payload, prevent concurrent duplicate tool execution, isolate same IDs across users, and prune safely.

## S2-02: HTTP and SSE chat

Implement:

```text
POST /mobile/chat
POST /mobile/chat/stream
```

Require Bearer principal, reject client authorization fields, validate IDs/timestamps/size/mode/context, reserve idempotency, and call canonical `Assistant.generate_reply()` or `Assistant.stream_reply()` with explicit `active_user_id`. Never use the GUI direct-LLM helper. Emit only canonical JSON/SSE events and status semantics.

## S2-03: Structured runtime events

Expose truthful tool call/result/approval/terminal events only from existing action/result boundaries. Do not parse prose to infer actions. Preserve permissions, policy, approval requirements, and verification evidence. Keep a capability false when the real event/approval path is unavailable.

## S2-04: Secure WebSocket

Add a validated Flask-compatible WebSocket dependency. Implement first-frame `auth`, immutable principal binding, pre-auth rejection, authentication timeout, frame-size/frequency limits, close codes 4401/4403/4408/4429, idempotency before `ack`, reconnect-safe result replay, and canonical snake_case events.

## S2-05: Voice upload

Require authentication. Enforce 15 MiB and 60 seconds. Sniff supported containers and require successful decode. Use private temporary files with guaranteed cleanup. Return `BACKEND_UNAVAILABLE` when STT dependencies/models are absent. Transcribe through existing STT, then call canonical Assistant with explicit identity. Preserve real status/evidence.

## S2-06: Protected TTS

Require authentication, validate text length and voice ID, use existing TTS, enforce timeout/output limits, return JSON base64 plus MIME type, avoid query-string text/private logs, and clean temporary artifacts.

## S2-07: Final capabilities

Turn features true only after implementation, focused tests, and runtime dependency checks. Keep live voice false. Keep unsupported domains 501/false. Home Assistant and approvals remain false unless the complete authorized/verified mobile path is implemented.

## S2-08: Shared mobile authentication changes

In the mobile repo:

- add session/capabilities constants;
- create a stable SecureStore device ID;
- send device metadata at login;
- restore through `/mobile/auth/session`;
- parse nested errors;
- keep local JWT decoding non-authoritative;
- add actual tests and CI/typecheck scripts.

## S2-09: Align draft PR #3

Change `authenticate` to `auth`; use nested client metadata and snake_case events; generate IDs with cryptographic primitives; handle documented close codes; replay exact pending payloads; share IDs/idempotency across WS and HTTP; reject malformed events; use `completed` for ordinary chat.

## S2-10: Align draft PR #5

Match snake_case voice/TTS fields, accept authenticated base64 audio, preserve limits/timeouts/cancellation, keep local checks as UX only, keep mocks explicit and `attempted`, and add contract tests.

## S2-11: Cross-repository contract tests

Create explicit test vectors in both repos for login/refresh/session errors, WS auth/ack/token/tool/approval/done/error events, HTTP/SSE chat, voice/TTS, API version, and capabilities. Tests must fail on field-casing drift or missing required fields.

## S2-12: Smoke and docs

Backend smoke: LAN server, status/capabilities, login/refresh, authenticated WS, normal canonical chat, duplicate WS/HTTP without duplicate execution, voice when available, TTS, and logout invalidation.

Physical-iPhone smoke when available: URL/login, restoration, chat/reconnect, microphone upload, base64 playback, logout/revocation, and background/foreground behavior. Record not-run items honestly.

## Session 2 tests

Cover HTTP/SSE/WS auth and event grammar, pre-auth rejection, close codes, ack-after-reservation, concurrent/cross-transport duplicates, two-user isolation, permission/approval preservation, no double tool execution, voice media/size/duration/decode/cleanup/unavailable behavior, TTS auth/voice/timeout/MIME/base64/cleanup, capability truth, log redaction, and mobile contract tests.

## Session 2 validation

Backend:

```powershell
pytest -q tests/mobile_api
pytest -q tests/test_assistant.py tests/test_assistant_identity_binding.py
pytest -q
ruff check .
black --check --diff rex/ tests/ bridge/ *.py
python -m compileall -q rex scripts
mypy rex --ignore-missing-imports
detect-secrets scan --baseline .secrets.baseline
python -m rex doctor
```

Mobile:

```powershell
npm ci
npm run lint
npx tsc --noEmit
npx expo-doctor
```

Add and run real test/export scripts before claiming them passed. Verify both worktrees clean. Open one backend PR and update the coordinated mobile draft PRs, link #323, wait for/fix all CI, do not merge, and report automated/LAN/physical results separately.

## Workstream closure

Close #323 only after backend and mobile PRs merge, contracts match, rotation/revocation/idempotency/isolation are proven, capabilities are truthful, LAN smoke passes, and physical-iPhone validation is complete or explicitly tracked as a remaining release blocker. Public exposure remains separately gated on TLS and production reverse-proxy work.
