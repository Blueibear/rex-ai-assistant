# Verification Report: Phase 5, Cycle 5.1 Windows Computer Control Client Foundation

## Scope
Audit the merged implementation claims for the Cycle 5.1 computers client foundation, validate with runnable commands, and apply minimal follow-up fixes for safety and documentation alignment.

## Claims vs Verified Reality

| Claim | Result | Evidence |
|---|---|---|
| New computers client foundation files exist | ✅ Verified | `rex/computers/__init__.py`, `rex/computers/config.py`, `rex/computers/client.py`, `rex/computers/service.py` present |
| CLI has `rex pc list/status/run` | ✅ Verified | `rex/cli.py` parser + `cmd_pc` implementation |
| Example config added | ✅ Verified | `config/rex_config.example.json` has `computers` section |
| `docs/computers.md` added | ✅ Verified | File exists and documents contract/config/commands |
| `CLAUDE.md` updated for computers keys/commands/safety | ✅ Verified (and adjusted) | Keys and commands exist; run command updated to require `--yes` |
| `tests/test_computers.py` has 44 offline tests | ✅ Verified | `pytest -q tests/test_computers.py` collected/passed 44 tests |
| Quality gates pass (tests/lint/format) | ✅ Verified for current branch | `pytest -q`, changed-file Ruff/Black, compileall |

## CI Lint/Format Scope (Ground Truth)

From `.github/workflows/ci.yml`, the `Lint & Format Check` job:
- computes base branch
- diffs changed `*.py` files only: `git diff --name-only "origin/$BASE_REF...HEAD" -- '*.py'`
- runs `ruff check` and `black --check --diff` on that changed-file set
- runs `python -m compileall -q rex scripts`

Local validation mirrored this changed-file model for touched Python files.

## Security Analysis

### 1) Allowlist enforcement before network

Verified in `ComputerService.run()`:
- resolves computer config
- checks `cfg.is_command_allowed(command)`
- raises `AllowlistDeniedError` before client creation and before HTTP.

Follow-up hardening test added:
- `test_disallowed_command_raises_before_network` now patches `svc._make_client` and asserts `assert_not_called()`, proving no client/network path is entered on deny.

### 2) Secrets handling and logs

Verified:
- config uses `auth_token_ref` indirection; tokens resolved via `CredentialManager.get_token(...)`.
- no token storage in runtime config model fields.
- logging in client uses safe labels and command name; header/token values are not logged.

### 3) URL handling and timeouts

Verified:
- config validator enforces `http/https` scheme and required netloc.
- client URL building uses `urljoin(self._base_url + '/', path.lstrip('/'))`.
- all outbound calls set timeout:
  - requests backend uses `(connect_timeout, read_timeout)` tuple
  - urllib backend uses bounded timeout value.

### 4) Approval/guardrail safety for `pc run`

Audit finding:
- prior implementation allowed immediate remote execution via CLI without policy-engine or confirmation guard.
- repo has strong approval patterns for other high-risk flows (`policy_engine`, approvals CLI), but `pc run` was not wired.

Mitigation implemented (smallest safe change):
- `rex pc run` now requires explicit `--yes` confirmation before execution.
- without `--yes`, command exits with a warning and does not initialize service/network path.
- CLI help/description and docs updated accordingly.
- tests added for both no-`--yes` refusal and `--yes` execution path.

## Correctness/UX Checks

- `rex pc list` prints configured computers with id/label/url/allowed commands and totals.
- `rex pc status` requests status and prints structured host fields.
- `rex pc run` builds command + args payload and returns output/exit code.

## Tests Offline and Deterministic

Verified:
- `tests/test_computers.py` uses in-process `HTTPServer` fixture on loopback and mocks (`MagicMock`).
- no external network dependency.
- tests did not modify tracked files (`git status --porcelain` clean before changes; clean after test runs aside from intentional edits).

## Discrepancies Found and Fixes Applied

1. **Safety gap**: `rex pc run` previously lacked explicit high-risk guard.
   - Fixed by requiring `--yes`.
2. **Proof strength gap**: allowlist-deny test did not explicitly prove no client/network creation.
   - Fixed by asserting `_make_client` is never called on deny.
3. **Docs drift**: command examples lacked new confirmation behavior.
   - Fixed in `docs/computers.md` and `CLAUDE.md`.

## Commands Executed and Results

```bash
git status --porcelain
python -m rex --help
pytest -q
pytest -q tests/test_computers.py
python scripts/security_audit.py
python -m compileall -q rex scripts
python -m ruff check rex/cli.py tests/test_computers.py
python -m black --check rex/cli.py tests/test_computers.py
```

All commands completed successfully in this environment. Security audit reported pre-existing placeholder findings in unrelated files and no exposed secrets.

## PR-ready Summary

- Verified all claimed Cycle 5.1 files/commands/tests are present and functioning.
- Confirmed allowlist-before-network, credential indirection, timeout behavior, and offline tests.
- Added minimal high-risk guard (`--yes`) for `rex pc run` pending full policy-engine integration in Cycle 5.2.
- Strengthened tests to prove deny-path avoids client/network call.
- Updated docs and CLAUDE guidance to match real behavior.
