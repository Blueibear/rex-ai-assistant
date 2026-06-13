# Security Policy

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

AskRex Assistant is still in early development.

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

## Security baseline

The repository is periodically scanned with `scripts/security_audit.py` for merge conflict markers, placeholder/incomplete code, and exposed secrets.

A triaged inventory of all current findings is maintained at:

- [docs/security/AUDIT-INVENTORY.md](docs/security/AUDIT-INVENTORY.md)

The inventory classifies every finding as `production-blocker`, `dev-only-documented`, or `false-positive` and links each production blocker to the User Story that resolves it.
