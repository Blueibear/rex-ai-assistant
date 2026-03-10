
---

## `docs/claude/INTEGRATIONS_STATUS.md`

```md
# Claude Reference: Integrations Status

This file is reference material.
Use it when a task touches integrations, capability claims, roadmap sequencing, or docs language about readiness.

## Status language rules
Use only these labels in docs when appropriate:
- Stub
- Beta
- Production-ready

Do not use stronger language than the actual implementation supports.

## Current repo-wide reality
Rex is not production-ready overall.
Some subsystems are meaningful and usable.
Some integration surfaces are real but partial.
Some claims in older docs are broader than the current implementation supports.

## Integration status snapshot
### Email
Status: Beta
Reality:
- Real IMAP read and SMTP send backend exists
- Multi-account foundation exists
- Account-aware routing exists
- Must still be described carefully unless current verification says otherwise

### Calendar
Status: Beta
Reality:
- ICS read-only backend exists
- Local file and HTTPS source support exist
- Do not imply full calendar write support

### Messaging
Status: Beta
Reality:
- Twilio send backend exists
- inbound SMS webhook/store exists
- optional and credential-gated

### Notifications
Status: Beta
Reality:
- dashboard notification store exists
- API endpoints exist
- SSE exists for real-time push
- retention cleanup scheduling exists
- do not imply full production hardening automatically

### Voice identity
Status: Beta or MVP-in-progress depending on the specific doc
Reality:
- scaffolding exists
- enrollment and calibration commands exist
- optional dependency model exists
- do not overstate it as universally production-ready

### Windows computer control
Status: Beta
Reality:
- client foundation exists
- Windows agent server exists
- approval and allowlist model exists
- boot-persistence and service-wrapper hardening remain roadmap work

### WordPress and WooCommerce
Status: Beta
Reality:
- read-only monitoring and write-gated paths exist in current repo docs and code history
- keep wording specific to verified commands and approval-gated writes
- do not imply broad CMS automation

### Home Assistant TTS
Status: Beta
Reality:
- optional notification channel exists
- disabled by default
- auth and SSRF hardening matter

## Known caution areas from the audit
Be extra careful in these areas:
- autonomous tool execution claims
- scheduler-triggered workflow execution claims
- replay claims
- Docker deployment guidance
- Windows quickstart wording
- production-ready stabilization claims

## Roadmap phase order
Recommended phase order:
1. Finish notification usability end to end
2. Finish Windows computer-control safety and ops hardening
3. Add WordPress and WooCommerce integration work in the planned sequence
4. Promote voice identity from scaffolding to MVP
5. Real-time notifications and remaining hardening work

## Docs rules for integrations
When editing integration docs:
- include a top-level Implementation Status section
- keep commands and config snippets accurate
- separate real behavior from planned behavior
- avoid capability wording that outruns code or verification
- if README limitations change, update the related integration docs in the same PR

## When to include this file in a task packet
Include this file only when the task touches:
- integration docs
- roadmap sequencing
- feature-readiness wording
- capability claims
- implementation status labels