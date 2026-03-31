# Verification Report: Cycle 5.3 Windows Agent Server

## Scope
This report verifies the Cycle 5.3 Windows agent claims against the current branch state by direct code inspection and local command execution.

## Claims vs verification

| Claim | Verification result | Evidence |
|---|---|---|
| New Flask agent server at `rex/computers/agent_server.py` | ✅ Verified | File exists; implements `/health`, `/status`, `/run`, token auth, allowlist, rate limit, timeout, truncation. |
| New runner `scripts/windows_agent.py` | ✅ Verified | File exists and delegates to `rex.computers.agent_server:main`. |
| New tests `tests/test_windows_agent.py` | ✅ Verified | File exists; expanded to 33 tests in this audit including OPTIONS auth coverage. |
| Updated docs `docs/computers.md` and `CLAUDE.md` | ✅ Verified | Both include Cycle 5.3 section and matching env vars. |
| `pyproject.toml` exposes `rex-agent` entry point | ✅ Verified | `[project.scripts] rex-agent = "rex.computers.agent_server:main"`. |
| `/health` requires auth and returns `{"status":"ok"}` | ✅ Verified | Valid token returns 200; missing/wrong token returns 401. |
| `/status` returns hostname/os/user/time and requires auth | ✅ Verified | Tests and code confirm required keys and auth. |
| `/run` allowlisted execution with `shell=False`, timeout, truncation | ✅ Verified | Code uses `subprocess.run(..., shell=False, timeout=...)`; allowlist check pre-exec; truncates stdout/stderr. |
| Constant-time token compare | ✅ Verified | `hmac.compare_digest` used. |
| Audit logging for `/run` without token leakage | ✅ Verified | `/run` logs remote/command/outcome only; token not logged. |
| CI gates (pytest, ruff, black, compileall, security_audit, doctor) | ⚠️ Partially reproduced | All requested commands were run; repo-wide `pytest -q` fails only due to repo-integrity tests detecting expected local modifications while auditing. |

## Security review notes

### 1) Mandatory auth for all endpoints
- `/health`, `/status`, `/run` enforce token auth.
- **Gap found/fixed**: Flask auto-OPTIONS requests were not explicitly covered by tests and could bypass route-level checks depending on method handling order.
- **Fix applied**: Added `@app.before_request` guard for all three API paths so auth/rate-limiting apply uniformly (including OPTIONS).

### 2) Token handling
- Uses `hmac.compare_digest` for constant-time token comparison.
- No token value appears in log templates or responses.
- Startup failure message references env-var name only, not token.

### 3) `/run` execution safety
- Rejects non-allowlisted commands before subprocess call.
- Uses `argv=[command]+args` and `shell=False`.
- Deny-path test verifies subprocess is not called.
- Args/quoting do not bypass allowlist because only exact `command` field is allowlist matched.

### 4) Allowlist parsing robustness
- Allowlist parser strips whitespace and ignores empty entries from comma-separated env value.
- Works cross-platform because parsing is string-based and command dispatch is explicit argv.

### 5) Rate limiting
- Fixed-window, per-IP in-memory limiter.
- `rate_limit<=0` disables limiting intentionally.
- Failure mode is deny (429) once limit reached.

### 6) Output truncation and timeout behavior
- Applies independently to both stdout and stderr byte streams.
- Safe decode with `errors="replace"` avoids decode crashes.
- Timeout returns deterministic JSON with `exit_code=-1` and timeout message.

## Environment/config surface audited

Agent env vars used by code:
- `REX_AGENT_TOKEN` (required)
- `REX_AGENT_TOKEN_ENV`
- `REX_AGENT_HOST` (default `127.0.0.1`)
- `REX_AGENT_PORT` (default `7777`)
- `REX_AGENT_ALLOWLIST` (default `whoami`)
- `REX_AGENT_RATE_LIMIT` (default `60` per minute)
- `REX_AGENT_TIMEOUT` (default `30`)
- `REX_AGENT_MAX_OUTPUT` (default `65536`)
- `REX_LOG_LEVEL` (startup logging level)

Docs and CLAUDE entries match this surface and defaults, including localhost-safe binding default.

## Client/contract integration check

- `rex/computers/client.py` contract matches documented paths/headers/body/response fields for `/health`, `/status`, `/run`.
- `ComputerService` client-side allowlist behavior remains unchanged (Cycle 5.1 compatibility preserved).
- Additional `duration_ms` field from agent is tolerated by client (client ignores extra keys).

## Commands run and outcomes

1. `python -m pip install -e ".[dev]"` ✅ pass
2. `pytest -q tests/test_windows_agent.py` ✅ pass (33 passed after audit additions)
3. `pytest -q` ⚠️ expected fail while working tree is modified (repo-integrity tests fail on modified tracked files)
4. `python -m ruff check rex/computers/agent_server.py tests/test_windows_agent.py` ✅ pass
5. `python -m black --check rex/computers/agent_server.py tests/test_windows_agent.py` ✅ pass
6. `python -m compileall -q rex scripts` ✅ pass
7. `python scripts/security_audit.py` ✅ pass (no exposed secrets; placeholder findings are pre-existing heuristic output)
8. `python scripts/doctor.py` ✅ pass with non-blocking environment warnings (ffmpeg/torch/REX_SPEAK_API_KEY)
9. `rex-agent` without token ✅ fails safely with startup error requiring token

## Gaps found and fixed

1. **Auth/rate-limit guard centralization**
   - Fixed by adding global `before_request` guard for `/health`, `/status`, `/run`.
2. **Missing OPTIONS auth coverage in tests**
   - Added tests verifying OPTIONS on protected endpoints requires valid token.

## Recommended next items (not in this patch)

1. Windows Service wrapper for durable startup/restart policy.
2. TLS/mTLS termination guidance and hardening profile for remote exposure.
3. Policy integration for per-command/per-arg constraints beyond name allowlist.
4. Persistent append-only audit sink (file/ETW/syslog) with rotation and integrity controls.
