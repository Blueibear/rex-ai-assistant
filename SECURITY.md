# Security Policy

## Security baseline

The current `scripts/security_audit.py` baseline and triage inventory is tracked in [docs/security/AUDIT-INVENTORY.md](docs/security/AUDIT-INVENTORY.md). Release validation must run `python scripts/security_audit.py --release-gate`; actionable findings and invalid or expired suppressions exit nonzero. Developer mode remains informative and does not substitute for the release gate.

CI enforces this on every push and pull request: the **Security Audit Gate** job in `.github/workflows/ci.yml` runs `python scripts/security_audit.py --release-gate` and fails the build on any nonzero exit. CI also rejects committed generated artifacts (`scripts/check_no_generated_artifacts.py`) and committed secrets (`detect-secrets` against `.secrets.baseline`, covering the whole tree including `config/`).

## Desktop credential storage

Packaged Windows builds store secrets only in the DPAPI-backed credential
vault. Non-secret config contains opaque references bound to scope, validated
Rex user, integration, authorized account, and credential slot. Packaged bridge
processes strip the legacy plaintext fallback flag. Vault corruption, context
mismatch, ACL-hardening failure, and persistence/readback failure fail closed.

The migration command is dry-run by default and requires explicit scope and
owner. Apply mode verifies encrypted storage and the reference registry before
atomically sanitizing plaintext sources; it creates no plaintext backup and
emits no secret-derived preview, length, or hash. See
[docs/credentials.md](docs/credentials.md).

## Mobile and public API boundary

The future `https://askrex.app` hostname is **not** permission to expose the existing local Flask/admin services. Public/mobile ingress may target only the dedicated `rex.mobile_api` `/mobile/*` contract through a path allowlist and loopback-only origin. `rex.gui_app`, the computer agent, TTS service, OpenClaw tool server, credentials, logs, and admin/configuration routes remain non-public trust zones.

Current paired LAN access uses desktop-owned S7 certificate fingerprint/SPKI binding. A tunnel terminating WebPKI TLS at an external edge must not disable or fake that control; public pairing stays gated until the versioned public transport-binding design is implemented and tested. See `docs/mobile/MOBILE_API_THREAT_MODEL.md` and `docs/mobile/ASKREX_APP_GATEWAY.md`.

Tunnel/API credentials must never be committed. The reference deployment in `docs/mobile/CLOUDFLARE_TUNNEL.md` uses placeholders only and keeps provider credential files outside the repository.

## Reporting a vulnerability

If you discover a security issue in AskRex Assistant, please do **not** open a public GitHub issue with exploit details.

Instead, report it privately by emailing: **security@askrex.app**

Please include as much detail as you safely can, such as:

- What part of AskRex is affected
- Steps to reproduce the issue
- Expected impact
- Relevant logs, screenshots, or error messages
- Your operating system and environment, if relevant

Please remove any private tokens, passwords, API keys, or personal data before sending logs or screenshots.

## Response expectations

AskRex is an early-stage open-source project, so response times may vary.

The maintainer will make a best effort to:

- Acknowledge valid reports
- Investigate the issue
- Avoid exposing sensitive details publicly before a fix is available
- Credit reporters when appropriate and requested

## Supported versions

AskRex Assistant is a release-candidate implementation under validation, not a signed public production release.

At this time, only the latest code on the main development branch is actively considered for security fixes.

## Security-sensitive areas

Security reports are especially helpful for issues involving:

- Leaked secrets, tokens, or credentials
- Unsafe handling of Home Assistant tokens or URLs
- Unsafe local network exposure
- Remote access or authentication problems
- Unsafe plugin or tool execution
- Command execution risks
- Private user data or memory storage
- Local file access
- Dependency vulnerabilities

## Public disclosure

Please allow reasonable time for investigation before publicly disclosing a vulnerability.

If you are unsure whether something is a security issue, email first.

## Change Log

- **US-020 (2026-06-24):** `rex/replay.py` stub result behavior resolved. `replay()` now raises
  `NotImplementedError("replay is not available in this build")` instead of returning a
  misleading `{"status": "stub"}` dict. No caller receives a false positive result.
