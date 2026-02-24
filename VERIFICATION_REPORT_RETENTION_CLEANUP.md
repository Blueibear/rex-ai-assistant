# Verification Report: Retention Cleanup Scheduling

## Scope and method
I audited the merged retention cleanup scheduling work by inspecting implementation, wiring, tests, docs/config, and by running the required quality gates.

## Claims vs. verified truth

### A) Scheduling semantics and safety defaults

1. **Dashboard cleanup does not require inbound features** — **Verified true**.
   - Dashboard scheduling is controlled by `notifications.dashboard.store.cleanup_schedule` and does not depend on `messaging.inbound.enabled`.
2. **Inbound cleanup requires `messaging.inbound.enabled=true`** — **Verified true**.
3. **`null` disables each schedule independently** — **Verified true**.
4. **`interval:86400` parser behavior in scheduler** — **Verified true**.
   - The scheduler parses `interval:<seconds>` and computes next runs from that integer interval.
5. **Idempotent registration and safe repeated execution** — **Verified true**.
   - Registration checks existing job IDs before adding.
   - Cleanup methods are DELETE-by-cutoff operations and repeated runs on same data are safe.

### B) Job naming and manual triggering

1. **Stable job IDs are used and documented** — **Verified true**.
   - `dashboard_retention_cleanup` and `inbound_sms_retention_cleanup` are used consistently in code/docs/tests.
2. **Manual trigger command behavior** — **Partially true before fix; fully true after fix**.
   - I found a defect: CLI `rex scheduler run <job_id>` initialized legacy scheduler jobs via `rex.integrations.initialize_scheduler_system()`, which previously did not register retention jobs. Result: `dashboard_retention_cleanup` could be missing.
   - **Fix applied:** retention registration added to `rex.integrations.initialize_scheduler_system()`.
   - After fix, `rex scheduler run dashboard_retention_cleanup` succeeds with default config.
   - `inbound_sms_retention_cleanup` exists only when inbound is enabled, as designed.

### C) Correctness of cleanup implementation

1. **Dashboard cleanup invokes store retention logic** — **Verified true** (`DashboardStore.cleanup_old()`).
2. **Inbound cleanup invokes store retention logic with retention days** — **Verified true** (`InboundSmsStore.cleanup_old()`).
3. **Fail-safe behavior** — **Verified true**.
   - Setup functions and job callbacks catch/log exceptions; startup does not crash.
   - Logging in store modules avoids message body/phone number leakage at INFO.

### D) Wiring correctness

1. **No new network endpoint exposure in Flask wiring** — **Verified true**.
   - `wire_retention_cleanup()` only wires scheduler jobs; no route registration.
2. **services.py non-Flask wiring** — **Verified true**.
3. **Double-registration safety (flask + services)** — **Verified true** via job-ID dedupe checks.

### E) Docs and config accuracy

1. `docs/notifications.md` and `docs/messaging.md` match key names and behavior — **Verified true**.
2. `config/rex_config.example.json` includes both `cleanup_schedule` keys — **Verified true**.
3. `CLAUDE.md` includes keys and manual command guidance — **Verified true**.

## Issues found and fixes

### 1) CLI manual scheduler run path did not ensure retention jobs existed
- **Severity:** functional mismatch with docs/manual-trigger guidance.
- **Root cause:** `cmd_scheduler run` calls `initialize_scheduler_system(start_scheduler=False)` from `rex.integrations`; that initializer only registered email/calendar jobs.
- **Fix:** add `_try_register_retention_jobs()` in `rex.integrations.initialize_scheduler_system()` so retention jobs are loaded from config in CLI run path too.
- **Regression test added:** `tests/test_cli_scheduler.py::test_cli_scheduler_run_retention_jobs`.

## Commands run and outcomes

1. `python -m pip install -e ".[dev]"` — **PASS**
2. `pytest -q` — **FAIL in working tree due to repo-integrity tests detecting intentional tracked edits**
   - Failing tests: `tests/test_repo_integrity.py`, `tests/test_repository_integrity.py`.
   - This is expected during active patching; they enforce a clean tracked working tree.
3. `python -m ruff check rex/integrations.py tests/test_cli_scheduler.py` — **PASS**
4. `python -m black --check rex/integrations.py tests/test_cli_scheduler.py` — **PASS**
5. `python scripts/security_audit.py` — **PASS** (no secrets; known placeholder findings reported by tool)
6. `python -m compileall -q rex scripts` — **PASS**
7. `python -m rex --help` — **PASS**
8. `python scripts/doctor.py` — **PASS with environment warnings** (ffmpeg/torch/env settings)

Additional verification:
- `python -m rex scheduler run dashboard_retention_cleanup` — PASS after fix.
- `python -m rex scheduler run inbound_sms_retention_cleanup` — expected FAIL in current config (`messaging.inbound.enabled=false`).

## Follow-ups

- Optional: if desired, add a dedicated CLI test that asserts inbound retention run command fails when inbound is disabled (for stronger explicitness).
