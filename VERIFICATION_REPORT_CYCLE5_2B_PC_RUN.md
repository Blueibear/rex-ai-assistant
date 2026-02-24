# Verification Report — Cycle 5.2b `rex pc run` Policy + Approvals

## Scope and method
I verified the squash-merged implementation of Cycle 5.2b by:
- inspecting the claimed files and CLI execution path,
- running the required command checklist,
- executing targeted tests for `pc_run` policy and computer CLI safety,
- reviewing approval persistence and token-handling behavior.

## Claims vs verified reality

### A) Merge correctness

1. **`rex/computers/pc_run_policy.py` exists and is used by CLI flow** — ✅ **Verified**
   - File exists and exports `check_pc_run_policy()` and `find_pending_or_approved_approval()`.
   - `rex/cli.py` imports and calls `check_pc_run_policy()` in `cmd_pc` for `pc run`.

2. **`rex/policy_engine.py` marks `pc_run` as HIGH risk and approval-required** — ✅ **Verified**
   - `DEFAULT_POLICIES` includes `ActionPolicy(tool_name="pc_run", risk=RiskLevel.HIGH, allow_auto=False)`.

3. **CLI flow is allowlist → policy/approval → `--yes` → run** — ✅ **Verified**
   - Client-side allowlist check (`service.get_command_allowed`) happens before any approval creation.
   - Non-allowlisted commands return immediately.
   - Policy decision and approval lookup/create happen next.
   - `--yes` guard remains mandatory even when approval is already approved.
   - Only then does `service.run(...)` execute.

4. **Approval storage pattern matches workflow approvals and avoids secrets** — ✅ **Verified**
   - Uses `WorkflowApproval` with `approval.save(...)` to JSON in `data/approvals` by default.
   - Approval summary contains `computer_id`, `command`, `args`, allowlist decision, and `initiated_by`.
   - No token fields are written.

5. **`.gitignore` minimality for this cycle** — ⚠️ **Partially verified / nuance**
   - The merge commit message states only `data/approvals/` and `data/workflows/` were added.
   - Current `.gitignore` includes additional pre-existing `data/*` artifact ignores (`data/cues/`, `data/notifications/`, `data/scheduler/`, `data/*.db`).
   - I did **not** broaden `.gitignore` further.

### B) Security and safety invariants

1. **No token secrets are stored or logged in approvals path** — ✅ **Verified**
   - Approval payload builder excludes token values.
   - CLI/policy logging messages include computer ID/command/reason but not credential material.
   - Client and agent docs/comments reinforce token non-logging behavior.

2. **Allowlist enforcement before network and before approval creation** — ✅ **Verified**
   - `cmd_pc run` performs allowlist check first and exits on deny.
   - `ComputerService.run()` independently re-checks allowlist before creating API client/network call.

3. **Server-side allowlist remains enforced** — ✅ **Verified**
   - Agent `/run` handler enforces server-side allowlist before subprocess execution.
   - Approval does not bypass agent allowlist.

4. **`--yes` required even with approved approval** — ✅ **Verified**
   - CLI explicitly blocks without `--yes` after approved approval path.
   - Verified by tests.

### C) Idempotency and UX

1. **Deterministic stable step_id** — ✅ **Verified**
   - `_command_step_id()` hashes JSON of `{cid, cmd, args}` with sorted keys.
   - Identical `(computer_id, command, args)` yields same step ID.
   - Different args produce different IDs.

2. **Args ordering behavior** — ✅ **Verified (intentional)**
   - Args list order is preserved in hash input; same ordered args => same approval lookup.
   - Different ordering intentionally yields a different approval request.

3. **Clear UX output around approvals and `--yes`** — ✅ **Verified**
   - Pending path prints approval ID and exact `rex approvals --approve/--deny` commands.
   - Re-run guidance is printed.
   - Missing `--yes` message clearly explains explicit confirmation requirement.

4. **Repeat runs reuse pending approval (no spam)** — ✅ **Verified**
   - `find_pending_or_approved_approval()` scans approval files for matching workflow+step and returns existing pending/approved record.
   - Targeted tests confirm single-file idempotency behavior.

### D) Repo hygiene and test isolation

1. **Do tests write tracked files during this verification?** — ✅ **Verified clean for executed tests**
   - `tests/test_pc_run_policy.py` uses `tmp_path` + monkeypatch for approval dir isolation.
   - `tests/test_computers.py` pc-run safety tests monkeypatch approval dir to temp dirs.
   - `git status --porcelain` remained clean before/after required command runs.

2. **Is `.gitignore` masking test behavior in this area?** — ✅ **No masking required for Cycle 5.2b tests**
   - The pc-run policy tests executed cleanly using temp dirs and did not rely on repo-local artifact paths.

## Required command outcomes (evidence)

1) Clean checkout / full suite / cleanliness:
- `git status --porcelain` → clean (no output).
- `pytest -q` → **fails** during collection with unrelated pre-existing error in `tests/test_voice_loop.py` (`AttributeError: 'NoneType' object has no attribute 'ndarray'` from `rex/voice_loop.py`).
- `git status --porcelain` after run → still clean (no tracked-file modifications).

2) Ruff + Black on requested files:
- `python -m ruff check rex/computers/pc_run_policy.py rex/policy_engine.py rex/cli.py tests/test_pc_run_policy.py tests/test_computers.py` → pass.
- `python -m black --check rex/computers/pc_run_policy.py rex/policy_engine.py rex/cli.py tests/test_pc_run_policy.py tests/test_computers.py` → pass.

3) Security tooling:
- `python scripts/security_audit.py` → pass (no exposed secrets found).

4) Smoke:
- `python -m rex --help` → pass.
- `python scripts/doctor.py` → pass with environment warnings (`ffmpeg`, `torch`, `REX_SPEAK_API_KEY`).

Additional targeted validation:
- `pytest -q tests/test_pc_run_policy.py tests/test_computers.py` → **69 passed**.

## Fixes made
No code changes were required for Cycle 5.2b policy+approval behavior based on the checklist and targeted validation.

## Deferred follow-ups (explicit)
1. **Unrelated full-suite blocker**: `pytest -q` fails on `rex/voice_loop.py` typing/import behavior in `tests/test_voice_loop.py`; outside Cycle 5.2b scope but should be fixed to restore full CI reliability.
2. **Historical `.gitignore` breadth**: repo currently ignores several `data/*` runtime paths beyond `approvals/workflows`; no new broadening in this verification, but a future cleanup could re-evaluate whether all entries remain necessary.
