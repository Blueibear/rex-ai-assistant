# Verification Report: Phase 6.1 WordPress + WooCommerce Read-Only Monitoring

## Scope
This report audits the currently merged Phase 6.1 implementation on branch `master` baseline and records verification/fixes performed on branch `fix/verify-phase6-1-wp-wc`.

## Claude claims vs verified state

| Claim | Verified? | Evidence | Notes |
|---|---|---|---|
| WordPress read-only health monitoring with optional auth check | Yes | `rex/wordpress/client.py`, `rex/wordpress/service.py`, `rex/cli.py` | `GET /wp-json` plus optional `GET /wp-json/wp/v2/users/me`. |
| WooCommerce read-only orders/products listing with low-stock filter | Yes | `rex/woocommerce/client.py`, `rex/woocommerce/service.py`, `rex/cli.py` | Uses `GET /wp-json/wc/v3/orders` and `GET /wp-json/wc/v3/products`; low-stock filter is client-side. |
| Pydantic v2 config models with strict fields | Yes | `rex/wordpress/config.py`, `rex/woocommerce/config.py` | `ConfigDict(extra="forbid")` and validators present. |
| Service facades use CredentialManager refs | Yes | `rex/wordpress/service.py`, `rex/woocommerce/service.py` | Uses `get_token(...)` for configured refs. |
| CLI commands implemented | Yes | `rex/cli.py` | `rex wp health`, `rex wc orders list`, `rex wc products list` present and wired. |
| 76 offline tests | Partially outdated | `tests/test_wordpress.py`, `tests/test_woocommerce.py` | Initially 76 tests passed; after security-hardening tests, targeted suite is now 84. |
| Docs/README/CLAUDE updates | Mostly yes, but incomplete security guidance | `docs/wordpress_woocommerce.md`, `README.md`, `CLAUDE.md` | Added missing SSRF and sanitized-error notes in this fix pass. |
| No heavy deps / no Pipfile change | Yes | `Pipfile`, `Pipfile.lock` checks | No dependency-file edits in this fix pass. |

## Findings and discrepancies

### 1) SSRF hardening gap (fixed)
- **Issue:** WordPress and WooCommerce clients accepted arbitrary `base_url` host targets with no SSRF-oriented host validation.
- **Risk:** Configurable URLs could target localhost/private/reserved addresses in environments where configs are supplied by less-trusted operators or automation.
- **Fix applied:** Added host validation in both clients to reject localhost/private/loopback/link-local/reserved/multicast/unspecified targets and reject embedded URL credentials.

### 2) Unsanitized exception exposure (fixed)
- **Issue:** Client code returned `str(exc)` into user-visible result errors and warning logs.
- **Risk:** Some exception messages can include request details and potentially credential-adjacent data.
- **Fix applied:** Added `_safe_error_message(...)` in WordPress/WooCommerce clients to normalize timeout/network/HTTP failures and avoid raw exception leakage.

### 3) Offline determinism with DNS-based SSRF checks (fixed in tests)
- **Issue:** With SSRF DNS resolution checks, tests must mock `socket.getaddrinfo` to stay deterministic and fully offline.
- **Fix applied:** Added autouse fixtures in WP/WC tests to mock DNS resolution and added security tests for localhost rejection and sanitized errors.

## Commands run and outcomes

### Required gates

1. `python -m pip install -e ".[dev]"`
- **Result:** Pass
- **Output (excerpt):** `Successfully installed ... rex-ai-assistant-0.1.0 ...`

2. `pytest -q`
- **Result:** **Expected fail in dirty working tree context**
- **Why:** Repo integrity tests fail whenever tracked files are modified before commit.
- **Output (excerpt):**
  - `FAILED tests/test_repo_integrity.py::TestRepoIntegrity::test_no_tracked_files_modified`
  - `FAILED tests/test_repository_integrity.py::test_no_tracked_files_modified`

3. `pytest -q tests/test_wordpress.py tests/test_woocommerce.py`
- **Result:** Pass
- **Output:** `84 passed in 0.51s`

4. `python -m rex --help`
- **Result:** Pass
- **Output (excerpt):** help lists `wp` and `wc` command groups with examples for:
  - `rex wp health --site myblog`
  - `rex wc orders list --site myshop`
  - `rex wc products list --site myshop --low-stock`

5. `python scripts/security_audit.py`
- **Result:** Pass
- **Output (excerpt):**
  - `CLEAN - No merge markers found`
  - `CLEAN - No exposed secrets found`

6. `python -m ruff check rex/wordpress/client.py rex/woocommerce/client.py tests/test_wordpress.py tests/test_woocommerce.py`
- **Result:** Pass (after one import-order auto-fix)

7. `python -m black --check rex/wordpress/client.py rex/woocommerce/client.py tests/test_wordpress.py tests/test_woocommerce.py`
- **Result:** Pass (after formatting test files)

8. `PIPENV_IGNORE_VIRTUALENVS=1 pipenv lock --clear`
- **Result:** Warning / not runnable in this environment
- **Output:** `bash: command not found: pipenv`

9. `rg -n '"torch"|"triton"|"nvidia-' Pipfile.lock || true`
- **Result:** Pass (no matches)

## Files changed in this verification/fix pass
- `rex/wordpress/client.py`
- `rex/woocommerce/client.py`
- `tests/test_wordpress.py`
- `tests/test_woocommerce.py`
- `docs/wordpress_woocommerce.md`
- `CLAUDE.md`
- `VERIFICATION_REPORT_PHASE6_1_WORDPRESS_WOOCOMMERCE.md` (this file)

## Rationale for fixes
- Preserve read-only behavior while reducing security risk and improving safe error handling.
- Keep tests deterministic/offline while validating new SSRF controls.
- Align docs and project instructions with actual implementation/security requirements.

## Deferred follow-ups (explicit)
- Optional: centralize URL SSRF validation into a shared helper for all HTTP integrations to avoid duplicated policy logic.
- Optional: decide project policy for allowing private/internal hosts when deployment requires intranet WordPress/WooCommerce; current behavior is strict-deny for safety.

## PR-ready description text

### Short summary
Audit-verified Phase 6.1 WP/WC read-only monitoring and fixed two security gaps: SSRF-sensitive `base_url` handling and unsanitized exception exposure. Added deterministic offline tests for these cases and updated docs/CLAUDE guidance accordingly.

### Risk notes
- Behavior change: WP/WC integrations now reject localhost/private/reserved target hosts and URLs with embedded credentials.
- If intranet/private-host monitoring is required, a future explicit allowlist mechanism may be needed.

### Verification commands
- `python -m pip install -e ".[dev]"`
- `pytest -q`
- `pytest -q tests/test_wordpress.py tests/test_woocommerce.py`
- `python -m rex --help`
- `python scripts/security_audit.py`
- `python -m ruff check rex/wordpress/client.py rex/woocommerce/client.py tests/test_wordpress.py tests/test_woocommerce.py`
- `python -m black --check rex/wordpress/client.py rex/woocommerce/client.py tests/test_wordpress.py tests/test_woocommerce.py`
- `PIPENV_IGNORE_VIRTUALENVS=1 pipenv lock --clear`
- `rg -n '"torch"|"triton"|"nvidia-' Pipfile.lock || true`
