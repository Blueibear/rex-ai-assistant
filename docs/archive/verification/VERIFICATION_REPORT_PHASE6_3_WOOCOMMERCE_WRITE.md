# Verification Report: Phase 6.3 WooCommerce Write Actions with Policy Approval

## Scope
Audit and verify the Phase 6.3 merge on current `master` for:
- `rex/woocommerce/write_policy.py`
- WooCommerce client write paths in `rex/woocommerce/client.py`
- service facade write routing in `rex/woocommerce/service.py`
- CLI write commands in `rex/cli.py`
- tests in `tests/test_wc_write_actions.py`
- docs alignment in `docs/wordpress_woocommerce.md` and `CLAUDE.md`

## A) Merge presence and repository state

### Commands
- `git status --porcelain`
- `git log -n 25 --oneline`
- `test -f rex/woocommerce/write_policy.py`
- `test -f tests/test_wc_write_actions.py`
- `git show --name-only --oneline 9a863df --`

### Findings
- Phase 6.3 is present in commit `9a863df` (`Claude/review woocommerce setup ly bz0 (#195)`), touching exactly the expected files.
- Required files exist (`write_policy.py` and `test_wc_write_actions.py`).

## B) Approval flow correctness and safety invariants

### Verified
- `workflow_id` is hardcoded to `"wc_write"`.
- Deterministic `step_id` is derived from a SHA-256 prefix over stable JSON keys: `action`, `site`, and `ids`.
- Approval lookup returns existing pending/approved approvals for the same deterministic step.
- Two-step behavior is implemented:
  1. policy/approval gating first,
  2. explicit `--yes` required even after approval.
- Approval summaries only contain action metadata (`site_id`, operation params, initiated_by).
- No consumer key/secret or auth tokens are stored in approval payloads.

### Security notes
- `check_wc_write_policy()` denies immediately on policy denial and does not create approvals for denied calls.
- Approval summaries are redacted by construction (no credential fields added by policy helper).

## C) CLI wiring and UX

### Verified behavior
- Commands exist and are wired:
  - `rex wc orders set-status ...`
  - `rex wc coupons create ...`
  - `rex wc coupons disable ...`
- First run without approval creates/returns pending approval and prints approval ID + approve/deny instructions.
- Re-run after approval still requires `--yes`.
- `--user` is threaded via `_resolve_wc_initiated_by()` into approval metadata.
- Error printing uses sanitized client/service messages.

### Discrepancy found and fixed
- **Issue:** top-level CLI help text still said WooCommerce was read-only.
- **Fix:** updated `rex/cli.py` user-facing help strings/docstrings to reflect approval-gated write actions.

## D) SSRF and URL hardening on write methods

### Verified
- Base URL validation rejects:
  - non-http(s) schemes (e.g., `file://`),
  - missing host,
  - embedded credentials,
  - localhost/localdomain,
  - private/loopback/link-local/reserved/multicast/unspecified resolved IPs.
- Same validation is used at client construction and therefore applies to read and write methods (`_get`, `_put`, `_post`).
- Write tests include SSRF-related coverage and have an autouse DNS mock (`socket.getaddrinfo`) for offline determinism.

## E) Write request construction

### Verified
- Endpoint/version paths match WooCommerce REST v3 under `/wp-json/wc/v3/...`.
- Methods and payloads are correct:
  - set order status: `PUT /orders/<id>` with `{status}`
  - add note: `POST /orders/<id>/notes` with `{note, customer_note}`
  - create coupon: `POST /coupons` with `code`, `amount`, `discount_type` (+ optional expiry/usage)
  - disable coupon: `PUT /coupons/<id>` with `{status: "draft"}`
- Tests assert exact URL and payload construction while mocking HTTP (`requests.put`/`requests.post`).

## F) Required command outcomes

- `python -m pip install -e ".[dev]"` ✅ pass
- `pytest -q` ✅ pass (initial full-suite baseline run)
- `python -m rex --help` ✅ pass
- `python scripts/security_audit.py` ✅ pass (no merge markers/secrets; placeholder findings are informational baseline)
- `python scripts/doctor.py` ✅ pass with expected environment warnings (`ffmpeg`, `torch`, `REX_SPEAK_API_KEY`)
- `python -m ruff check rex/woocommerce/write_policy.py rex/woocommerce/client.py rex/woocommerce/service.py rex/cli.py tests/test_wc_write_actions.py` ✅ pass
- `python -m black --check rex/woocommerce/write_policy.py rex/woocommerce/client.py rex/woocommerce/service.py rex/cli.py tests/test_wc_write_actions.py` ✅ pass

Additional verification after local fix:
- `python -m rex wc --help` ✅ pass (help now reflects write actions)

Note on one intermediate run:
- A post-fix `pytest -q` run failed *only* in repo-integrity tests because `rex/cli.py` was intentionally modified and uncommitted at that moment (dirty tracked file check). This is expected behavior of those tests, not a product defect.

## G) Dependency and lock rules

### Verified
- `Pipfile` was not modified in this audit.
- No ML/CUDA dependency additions were introduced.

### pipenv lock step
- `PIPENV_IGNORE_VIRTUALENVS=1 pipenv lock --clear` could not run because `pipenv` is not installed in this environment.

## Claims vs verified truth summary

| Claim | Result | Notes |
|---|---|---|
| New policy+approval module exists and mirrors pc_run design | ✅ Verified | Deterministic step-id, workflow id, pending/approved lookup, no secrets in summary. |
| Client write methods and helpers added with sanitization | ✅ Verified | `_put`, `_post`, `WriteResult`, and sanitized error handling present. |
| Service write facade integrates credentials and safe failures | ✅ Verified | Uses `CredentialManager`; raises explicit missing/site errors. |
| CLI write commands + approval + `--yes` + `--user` wiring | ✅ Verified | Full two-step gate behavior present. |
| Tests/docs updates are present and correct | ✅ Verified with one fix | 62 write tests exist; docs updated; CLI help text mismatch fixed in this audit. |

## Final assessment
Phase 6.3 implementation is present, functionally correct, and security controls are in place (approval gating + SSRF hardening + sanitized errors). One user-facing help-text inconsistency was corrected.
