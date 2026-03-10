
---

## `docs/claude/CONFIG_AND_SECURITY.md`

```md
# Claude Reference: Config and Security

This file is reference material.
Use it when a task touches config loading, secrets, env vars, security controls, network exposure, or packaging.

## Core config split
- Runtime settings belong in `config/rex_config.json`
- Secrets belong in `.env` or credential storage already supported by the repo
- Never commit secrets
- Never move secrets into source-controlled runtime config

## Security defaults
- Prefer least privilege
- Treat all external inputs as untrusted
- Prefer localhost binding unless explicitly configured otherwise
- Anything that binds to a port must be authenticated and rate limited
- Network-facing features must fail safely when unconfigured

## Packaging and dependency policy
- Keep heavy ML, CUDA, or similar dependencies out of default installs unless explicitly justified
- Heavy optional functionality belongs in optional extras
- Runtime imports for optional packages must be guarded
- Dependency changes that affect packaging or lockability require explicit validation

## GPU install truth
- Supported GPU installs are requirements-file based
- Do not reintroduce GPU extras such as `.[gpu-cu118]`, `.[gpu-cu121]`, or `.[gpu-cu124]` unless they are fully working with the required wheel index behavior
- Keep GPU guidance aligned across docs and requirements files

## Credential rules
- Use the repo’s credential-manager path where applicable
- Tokens, passwords, and secrets must never be logged
- Approval records and audit records must not persist secrets

## Identity config truths
- Identity is session-scoped
- `--user` flag overrides session or runtime defaults
- Session state belongs in OS-appropriate temp storage, not in the repo

## Email config reference
- Runtime config lives under `email`
- Multi-account routing uses `email.default_account_id` and `email.accounts[]`
- Notification routing can use `email_account_id`
- Non-secret server values stay in runtime config
- Secrets stay in `.env` or credential storage

## Calendar config reference
- `calendar.backend` is `stub` or `ics`
- ICS source can be a local path or HTTPS URL
- Keep ICS guidance truthful as read-only unless code changes that reality

## Messaging config reference
- `messaging.backend` is `stub` or `twilio`
- Inbound SMS webhook behavior is optional and guarded by config
- Inbound store path and retention are runtime config, not secrets

## Notification dashboard config reference
- Notification dashboard store is SQLite-backed
- Retention and cleanup schedule belong in runtime config
- SSE is in-process and should not be described as cross-worker capable unless that changes

## Windows computer control config reference
- Remote computer entries live in `computers[]`
- Auth tokens must be resolved via credential references
- Allowlists are enforced client-side before network calls
- Server-side allowlists are defense in depth, not a substitute for client-side rules

## Windows agent env vars
- `REX_AGENT_TOKEN` required
- `REX_AGENT_HOST`
- `REX_AGENT_PORT`
- `REX_AGENT_ALLOWLIST`
- `REX_AGENT_RATE_LIMIT`
- `REX_AGENT_TIMEOUT`
- `REX_AGENT_MAX_OUTPUT`

Do not document public exposure as the default. Localhost first.

## WordPress and WooCommerce config reference
- Site entries live under `wordpress.sites[]` and `woocommerce.sites[]`
- Base URLs must be `http(s)`
- Base URLs must not embed credentials
- SSRF protections apply
- Credentials must be indirect references, not inline secrets

## Home Assistant TTS config reference
- Channel is optional and disabled by default
- Token must come from credential lookup
- Do not log tokens
- SSRF hardening and URL safety checks apply
- `allow_http` should stay false by default in production-oriented docs

## Security audit rules
Run:
```bash
python scripts/security_audit.py