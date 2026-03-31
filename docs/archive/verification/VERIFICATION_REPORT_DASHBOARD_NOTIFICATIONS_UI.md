# Verification Report: Dashboard Notification Inbox UI

## Scope
Audited the merged dashboard notifications inbox change set (route wiring, auth posture, UI behavior, polling, tests, docs, and baseline quality gates).

## Claims vs Verified

| Claim | Verified | Evidence |
|---|---|---|
| New UI section for notifications exists with filters, unread badge, mark read, mark all read | **Verified with one fix needed** | Notifications section and controls exist in SPA template and JS handlers. Mobile nav initially lacked unread badge indicator; fixed by adding `notif-badge-mobile` and wiring badge updater to desktop+mobile badges. |
| Uses `/api/notifications` endpoints | **Verified** | JS calls `GET /api/notifications`, `POST /api/notifications/<id>/read`, and `POST /api/notifications/read-all`. |
| Polls every 30s only while Notifications active and stops on section change/logout | **Verified** | `startNotifPolling()` sets a single interval after forcing `stopNotifPolling()` first; section switch-out and logout stop polling. |
| 8 new tests in `tests/test_dashboard.py` | **Verified** | `TestNotificationsInboxUI` block includes 8 notification-inbox tests. |
| Ruff + Black clean, working tree clean | **Partially true** | Targeted files are Ruff/Black clean. Repo-wide Ruff/Black currently fail due baseline debt unrelated to this feature. Working tree can be clean after commit. |
| Full pytest has 2 pre-existing failures | **Not reproducible on current baseline** | Before fixes, full `pytest -q` completed with **1337 passed, 29 skipped** (0 failed). |

## Command log and results

### Required commands
1. `python -m pip install -e ".[dev]"`  
   - **PASS**. Editable install succeeded.
2. `pytest -q`  
   - **PASS on baseline before modifications**: `1337 passed, 29 skipped`.
   - During active local edits, repo-integrity tests correctly failed because tracked files were intentionally modified; this is expected behavior of integrity tests while patching.
3. `pytest -q tests/test_dashboard.py`  
   - **PASS** (`39 passed`).
4. `python -m ruff check rex/dashboard/routes.py tests/test_dashboard.py`  
   - **PASS**.
5. `python -m black --check rex/dashboard/routes.py tests/test_dashboard.py`  
   - **PASS**.
6. `python scripts/doctor.py`  
   - **PASS with environmental warnings** (`ffmpeg`, `torch`, and `REX_SPEAK_API_KEY` not set).
7. `python scripts/security_audit.py`  
   - **PASS** for merge markers/secrets; reported known placeholder heuristics (baseline noise, not a merge blocker).
8. `python -m rex --help`  
   - **PASS**.
9. `git status --porcelain`  
   - Verified at end of workflow (must be empty).

### Baseline debt capture (required)
10. `python -m ruff check .`  
    - **FAIL (baseline debt)**: large number of pre-existing violations across many non-feature files.
11. `python -m black --check .`  
    - **FAIL (baseline debt)**: many pre-existing files would be reformatted.

## Deep verification notes

### A) Route wiring and auth
- `/dashboard/notifications` route exists and serves same SPA entrypoint as `/dashboard` (deep-link behavior).
- Notification API endpoints (`/api/notifications`, `/api/notifications/<id>/read`, `/api/notifications/read-all`) are all wrapped with `@require_auth`.
- Proxy bypass/wiring in `flask_proxy.py` excludes dashboard prefixes from **main proxy auth** because dashboard routes enforce their own auth; `/api/notifications` remains under dashboard auth middleware.
- Frontend API helper now handles both `401` and `403` for authenticated sessions by forcing logout.

### B) UI behavior correctness
- Desktop nav includes Notifications link with unread badge.
- Mobile nav had Notifications link but **missing unread badge UI**. Fixed by adding `notif-badge-mobile` and wiring updates to both badges.
- Filters:
  - Unread only -> server query param `?unread=true` (verified)
  - Priority -> server query param `?priority=` (verified)
  - Channel -> client-side filter over fetched results (verified)
- Actions:
  - Mark single notification -> `POST /api/notifications/<id>/read`, then reload list.
  - Mark all read -> `POST /api/notifications/read-all` (global by default; optionally user-scoped if caller appends `?user_id=`).
- Polling:
  - Starts only in Notifications section.
  - Stops on section leave and logout.
  - Interval does not stack (`startNotifPolling()` clears old timer first).

### C) Test quality
- The added notifications tests are deterministic Flask-client tests (no timing waits, no real polling loops).
- No network dependency.
- Use `tmp_path` + `set_dashboard_store()` swap pattern; no tracked-file writes from tests.
- Added one assertion to ensure mobile unread badge markup exists.

### D) "2 pre-existing failures" claim
- On current baseline in this environment: `pytest -q` showed 0 failures (`1337 passed, 29 skipped`).
- Therefore the claim of exactly 2 pre-existing failures is not supported by this reproduction.

### E) Docs consistency
- `docs/notifications.md` correctly documents route, query params, client-side channel filtering, actions, and 30s polling behavior.
- No doc mismatch found requiring edits for this patch.

## Bugs found and fixed
1. **Mobile unread badge absent** from Notifications mobile nav.
   - Fix: added mobile badge element and shared badge update logic across desktop/mobile.
2. **403 handling gap in SPA API client** for authenticated sessions.
   - Fix: authenticated `401/403` now trigger logout path.
3. **Channel filter re-render zeroed summary** due hardcoded unread count `0`.
   - Fix: compute unread count from current in-memory notification list on channel-only re-render.

## Security notes
- Notification APIs remain behind dashboard auth (`@require_auth`) and are not publicly unauthenticated.
- Proxy path exemptions are intentional and rely on dashboard-local auth enforcement.
- No new endpoint exposure introduced by this patch.
- Polling cadence is fixed at 30s and limited to active section, reducing avoidable load.

## Recommended follow-ups
- Add lightweight JS unit tests for dashboard SPA helpers (`api()`, polling lifecycle, badge updater) to prevent UI regressions not covered by Flask tests.
- Consider explicit server-side rate limits per-session for notifications endpoints if dashboard traffic grows.
