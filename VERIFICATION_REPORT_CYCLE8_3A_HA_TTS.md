# Verification Report — Cycle 8.3a Home Assistant TTS

## Claimed work summary
Phase 8.3a claims to add a Home Assistant TTS notification channel (`rex/ha_tts`), wire it into notifications, provide a CLI test command, add offline tests, and document configuration and security posture.

## Hard gate 1: remote/base reality check
Commands run:

```bash
git remote -v
git fetch origin --prune
git switch master
git pull --ff-only
```

Observed reality:
- `origin` remote is not configured in this environment (`fatal: 'origin' does not appear to be a git repository`).
- Only local branch present at start was `work`.
- Because no authenticated remote exists here, base/merge verification against `origin/master` could not be completed in-container.
- Current branch pointed to commit:
  - `c3f8272 feat(notifications): implement Home Assistant TTS notification channel (Cycle 8.3a) (#202)`

## Hard gate 2: lint/format discipline
Executed before readiness claims:

```bash
python -m ruff check rex/ha_tts rex/notification.py rex/cli.py tests/test_ha_tts.py
python -m black --check rex/ha_tts rex/notification.py rex/cli.py tests/test_ha_tts.py
```

Result: clean after fixes.

## Verification findings by area

### A) HA TTS package existence and behavior
Verified present:
- `rex/ha_tts/config.py`
- `rex/ha_tts/client.py`
- `rex/ha_tts/__init__.py`

Verified behavior:
- Disabled channel returns `None` from `build_ha_tts_client()`.
- Missing `base_url` or `token_ref` returns `None` with warnings.
- CredentialManager token lookup via `token_ref` is used (`CredentialManager().get_token(cfg.token_ref)`).
- No token persisted in runtime config structures.

Discrepancy found and fixed:
- Config model was permissive (`extra="ignore"`) and not strict.
- Updated to strict parse rules (`ConfigDict(extra="forbid", strict=True)`).

### B) Notification wiring
Verified `_send_to_ha_tts()`:
- Uses `build_ha_tts_client()` and `client.speak()`.
- Disabled or missing config path is silent no-op (no exception).
- Failed `speak()` raises `RuntimeError` to preserve retry/fallback behavior.

Discrepancy found and fixed:
- Metadata overrides `ha_tts_domain`/`ha_tts_service` were passed in payload data instead of affecting service path.
- Fixed by applying per-notification overrides to `client.tts_domain`/`client.tts_service` for that call only, then restoring originals.

### C) CLI behavior
Verified:
- `rex ha tts test` is wired and appears in root help and subcommand help.
- Default output clearly reports disabled state without network calls.
- `--message` and `--entity-id` arguments exist and are used.
- Incomplete config is handled safely; no token emitted in CLI output.

### D) SSRF mitigation
Validated in code and tests:
- Rejects non-http(s) schemes unless `allow_http=True`.
- Rejects embedded URL credentials.
- Rejects hosts resolving to loopback/private/link-local/reserved/multicast/unspecified addresses.
- Tests mock `socket.getaddrinfo` and `requests.post` to remain offline and deterministic.

### E) Docs/config/CLAUDE alignment
Verified:
- `config/rex_config.example.json` contains `notifications.ha_tts` block with defaults.
- `docs/home_assistant.md` documents keys, credential flow via `token_ref`, SSRF posture, CLI usage, and offline test guidance.
- `CLAUDE.md` includes HA TTS config keys, CLI commands, offline DNS-mocking testing expectations, conflict/remote verification rules, and Ruff+Black preflight gating.

### F) Dependency hygiene
Verified:
- No heavy dependency additions in `Pipfile` or `pyproject.toml` for this audit/fix.
- Existing `requests` dependency already present; no new HA-specific heavy deps introduced.

## Commands run and outcomes

### Install and quality gates
- `python -m pip install -e ".[dev]"` ✅
- `python -m ruff check rex/ha_tts rex/notification.py rex/cli.py tests/test_ha_tts.py` ✅
- `python -m black --check rex/ha_tts rex/notification.py rex/cli.py tests/test_ha_tts.py` ✅

### Tests
- `pytest -q tests/test_ha_tts.py` ✅ (46 passed)
- `pytest -q` ✅ after committing final changes on clean tree (integrity checks pass when repo is clean)

### Smoke/tooling
- `python -m rex --help` ✅
- `python -m rex ha tts test --help` ✅
- `python -m rex ha tts test` ✅ (reports disabled)
- `python scripts/security_audit.py` ✅ (no merge markers/secrets; non-blocking placeholder findings)
- `python scripts/doctor.py` ✅ (warnings for optional env/tools)
- `python -m compileall -q rex scripts` ✅

## Security review notes
- SSRF controls are implemented at URL validation time before HTTP calls.
- DNS resolution is validated against non-routable and sensitive IP classes.
- Token handling is via `CredentialManager` and private client attribute; no intentional token logging.
- Error strings are sanitized via `_safe_error()` to avoid leaking tokens or full request internals.

## Follow-ups / deferred items
- Environment lacks configured authenticated `origin`; maintainer should run remote/base checks in a fully configured local clone to prove merge ancestry against `origin/master`.
