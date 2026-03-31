# Verification Report: Phase 8.1 "Real-Time Notification Push" (SSE)

## Scope
This audit verified the merged SSE notification push implementation against claimed behavior, security requirements, and offline deterministic testability.

## Claims vs verified reality

### 1) `rex/dashboard/sse.py`
- **Claimed:** broadcaster existed with publish/subscribe, cleanup, keep-alives, singleton, test knobs.
- **Verified initially:** file did not exist.
- **Fix:** implemented a thread-safe in-process broadcaster with:
  - bounded per-subscriber queues,
  - non-blocking publish,
  - stale/dead subscriber cleanup,
  - keep-alive comment emission,
  - singleton `get_broadcaster()` / `set_broadcaster()`.

### 2) `GET /api/notifications/stream`
- **Claimed:** endpoint existed with auth, optional query token, init event, guarded test knobs, SSE headers.
- **Verified initially:** endpoint did not exist.
- **Fix:** added endpoint with:
  - `@require_auth` protection,
  - initial `init` event containing `unread_count`,
  - test-only knobs (`max_events`, `timeout`) honored only under `current_app.testing`,
  - SSE headers (`Content-Type`, `Cache-Control`, `Connection`, `X-Accel-Buffering`).

### 3) `DashboardStore.write()` publish hook
- **Claimed:** write path published SSE event best-effort.
- **Verified initially:** no publish hook.
- **Fix:** added best-effort publish in `write()` after DB persist; exceptions are caught and logged at debug level so writes never fail due to stream subscribers.

### 4) Dashboard UI EventSource-first
- **Claimed:** EventSource first with polling fallback and cleanup.
- **Verified initially:** polling only.
- **Fix:** added EventSource-first behavior in dashboard JS with automatic fallback to polling on stream error and stream cleanup on section switch/logout.

### 5) SSE tests
- **Claimed:** `tests/test_notification_sse.py` existed with full coverage.
- **Verified initially:** file did not exist.
- **Fix:** added comprehensive offline deterministic tests for broadcaster behavior, endpoint auth and format, query-token safety gates, user-scoped filtering, and store-write-to-SSE integration.

### 6) Docs
- **Claimed:** docs and CLAUDE updated.
- **Verified initially:** no SSE docs.
- **Fix:** updated `docs/notifications.md` and `CLAUDE.md` to document SSE behavior and security constraints.

## Security analysis

### A) Token query parameter risk
- Query-token auth was added only for `/api/notifications/stream` and only in safer contexts:
  - HTTPS (`request.is_secure` or `X-Forwarded-Proto=https`) OR localhost loopback.
  - same-origin when `Origin` is present.
- Unsafe remote plain-HTTP query token usage is rejected.
- Docs explicitly warn about URL leakage risk (history/intermediary logs) and recommend cookie/session auth when possible.

### B) Auth coverage
- `/api/notifications/stream` is protected with `@require_auth`.
- Existing Flask proxy bypass list (`/api/notifications`) still routes to dashboard auth, and the endpoint itself enforces auth.
- Added tests for unauthenticated denial.

### C) DoS / resource safety
- Subscriber queues are bounded (`max_events`) to prevent unbounded memory growth.
- Publisher is non-blocking and drops oldest event when a subscriber is saturated.
- Dead/closed subscribers are removed.
- Keep-alive cadence uses timeout/interval checks and does not busy-loop.
- Store write path is resilient even if publish fails.

### D) Data exposure and cross-user isolation
- Initial implementation had a multi-user leakage risk: global notifications API scope and no SSE user filtering.
- Fixes:
  - notifications list/read-all now default to authenticated session user and reject mismatched `user_id` scope,
  - mark-read is user-scoped,
  - SSE stream filters events to current authenticated user only,
  - login sessions now bind to active user key so per-user scoping is meaningful.

## Commands run and outcomes

- `python -m pip install -e ".[dev]"` ✅
- `python -m ruff check rex/dashboard/sse.py rex/dashboard/routes.py rex/dashboard_store.py tests/test_notification_sse.py` ✅
- `python -m black --check rex/dashboard/sse.py rex/dashboard/routes.py rex/dashboard_store.py tests/test_notification_sse.py` ✅
- `pytest -q tests/test_notification_sse.py` ✅
- `pytest -q tests/test_dashboard.py::TestNotificationsInboxUI::test_mark_all_read_end_to_end tests/test_dashboard.py::TestNotificationsInboxUI::test_list_notifications_with_unread_filter tests/test_dashboard.py::TestNotificationsInboxUI::test_list_notifications_with_priority_filter tests/test_notification_sse.py` ✅
- `python scripts/security_audit.py` ✅ (script reported existing repository placeholder findings; no secrets/merge markers)
- `python scripts/doctor.py` ✅ (environment warnings only: ffmpeg/torch/env vars)
- `python -m compileall -q rex scripts` ✅
- `pipenv lock --clear` ⚠️ not run (`pipenv` not available in this environment)

## Follow-ups / residual risks
- Query-token authentication still carries inherent URL exposure risk even with safety gates; prefer cookie/session auth in production reverse-proxy setups.
- SSE broadcaster is process-local; multi-process deployment would require broker-backed fanout to maintain consistency.
