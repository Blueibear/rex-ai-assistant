# Verification Report: Inbound SMS Webhook Wiring Audit

## Scope
Audit target: merged claim **"Inbound SMS webhook registration in flask_proxy with rate limiting + doctor readiness check + auth bypass for /webhooks/"**.

Audit date: 2026-02-22 UTC
Branch used for fixes: `fix/verify-inbound-webhook-wiring`

## Merged commit verification
Confirmed commit exists locally on master history:
- `0093b0b feat(messaging): register inbound SMS webhook in flask_proxy with rate limiting + rat… (#176)`
- File list in merged commit matches claim:
  - `rex/messaging_backends/webhook_wiring.py`
  - `rex/messaging_backends/inbound_webhook.py`
  - `rex/messaging_backends/inbound_store.py`
  - `flask_proxy.py`
  - `scripts/doctor.py`
  - `tests/test_inbound_webhook_wiring.py`
  - `config/rex_config.example.json`
  - `docs/messaging.md`
  - `CLAUDE.md`

## Claims vs. confirmed reality

### A) Repo state verification
- ✅ All claimed files exist in repo and contain matching responsibilities.
- ✅ Clean working tree was confirmed before fix work.
- ✅ Clean working tree confirmed again after tests and after this doc fix.

### B) Functional verification: webhook wiring
- ✅ `flask_proxy.py` creates the production Flask app and calls `register_inbound_sms_webhook(app)` at startup.
- ✅ Registration is config-gated: `messaging.inbound.enabled == false` returns early and route is not registered.
- ✅ Endpoint path is exactly `POST /webhooks/twilio/sms`.
- ✅ Tests confirm route registered when enabled and absent when disabled or token missing.

### C) Security verification
- ✅ Proxy auth bypass is narrowly scoped: only path prefix `"/webhooks/"` is bypassed from proxy-token/CF auth in `before_request`.
- ✅ Twilio signature verification is enforced by default in webhook blueprint:
  - Missing or invalid signature returns 403.
  - Uses constant-time signature comparison in `twilio_signature.py`.
- ✅ Reverse-proxy URL parity requirement is implemented/documented as `request.url` dependent verification.
- ✅ No secrets are logged by webhook wiring/validation paths.
- ✅ Rate limiting is applied when Flask-Limiter is available.
- ✅ If limiter init fails or Flask-Limiter is unavailable, behavior is graceful (route still works, warning/debug log, no crash).
- ✅ SSRF risk review: webhook handler performs no outbound HTTP calls; input is parsed and stored only.
- ✅ Arbitrary record write without valid signature is prevented when signature verification is enabled (default wiring path).

### D) Config and docs accuracy
- ✅ `config/rex_config.example.json` includes `messaging.inbound.rate_limit`.
- ✅ `CLAUDE.md` documents `messaging.inbound.rate_limit` and reverse-proxy expectations.
- ⚠️ Found docs gap in `docs/messaging.md`: inbound config JSON example + key list omitted `rate_limit` even though implementation supports it.
- ✅ Fixed docs gap in this branch by adding `rate_limit` to example and key list.

### E) Doctor and ops verification
- ✅ Doctor reports PASS when inbound webhook is disabled.
- ✅ Doctor check logic reports WARN with actionable hint when enabled but auth token missing.
- ✅ Doctor check reports PASS when enabled and auth token resolves (covered by tests).
- ✅ Doctor output does not print secrets.

### F) Tests
- ✅ `pytest -q` passed fully in this environment.
- ✅ `pytest -q tests/test_inbound_webhook_wiring.py` passed (14 tests).
- ✅ No pre-existing failing tests observed in current master state during this run.

### G) Lint/format gate verification
- ✅ Verified CI lint scope: changed-Python-files-only (from `.github/workflows/ci.yml`), not repo-wide.
- ❌ `python -m ruff check .` fails repo-wide on pre-existing issues (628 findings outside this change scope).
- ❌ `python -m black --check .` fails repo-wide on pre-existing formatting drift (98 files).
- ✅ This audit change touched docs-only and introduced no new Python lint/format regressions.

### H) Dependabot and lock safety
- ✅ No changes to `Pipfile` / `Pipfile.lock` in this audit fix.
- ⚠️ `pipenv` is not installed in this environment, so `pipenv lock --clear` could not be executed.
- ✅ No heavy ML/CUDA dependencies were introduced by this audit fix.

## End-to-end request path trace (`/webhooks/twilio/sms`)
1. Request enters `flask_proxy.py` app.
2. `before_request` bypasses proxy auth only for `"/webhooks/"` prefix.
3. Route exists only if startup wiring registered blueprint (requires inbound enabled + token + store init success).
4. Route handler in `inbound_webhook.py`:
   - Reads `X-Twilio-Signature`.
   - Reconstructs signed payload using `request.url` + form fields.
   - Validates signature via `validate_twilio_signature(...)`.
   - On failure -> `403 Forbidden`.
   - On success -> extracts `MessageSid`, `From`, `To`, `Body`.
   - Routes by `To` number -> matching `messaging.accounts[].from_number`.
   - Persists `InboundSmsRecord` to inbound SQLite store.
   - Returns empty TwiML `<Response/>` with HTTP 200.
5. If limiter is active, per-route limit applies before handler completion and can return HTTP 429.

## Commands run (in required order) and outcomes
1. ✅ `python -m pip install -e ".[dev]"` — success.
2. ❌ `python -m ruff check .` — failed repo-wide on pre-existing issues not introduced by this audit change.
3. ❌ `python -m black --check .` — failed repo-wide on pre-existing formatting drift.
4. ✅ `pytest -q` — `1235 passed, 29 skipped`.
5. ✅ `python scripts/doctor.py` — success; inbound webhook reported PASS (disabled by config).
6. ✅ `python scripts/security_audit.py` — success; no exposed secrets (placeholder findings are pre-existing).
7. ✅ `python -m compileall -q rex scripts` — success.
8. ⚠️ `PIPENV_IGNORE_VIRTUALENVS=1 pipenv lock --clear` — could not run (`pipenv: command not found`).
9. ✅ `git status --porcelain` — clean at checkpoints.

## Fixes applied in this audit
1. `docs/messaging.md`
   - Added missing `messaging.inbound.rate_limit` to inbound config JSON example.
   - Added missing `messaging.inbound.rate_limit` bullet in inbound key list.

Rationale: align docs with actual implemented and shipped config behavior.

## Final status
- Core merged functionality is present and works as described.
- Security posture for webhook auth/signature path is broadly correct for intended model.
- One documentation inaccuracy was found and fixed in this branch.
