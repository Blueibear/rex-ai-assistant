# Historical PRD Reconciliation

## Purpose

This inventory preserves useful requirements from the uploaded AskRex PRDs without allowing stale checkboxes, superseded architectures, or conflicting non-goals to override the authoritative build specification and active checklist.

The review covered:

- `PRD.md`, `PRD2.md`, `PRD3.md`, and `PRD4.md`
- completion, next-cycle, full-roadmap, and production-readiness PRDs
- GUI/autonomy/integration and voice-selector PRDs
- OpenClaw pivot and HTTP-integration PRDs
- repo-audit, test/fix, CI-fix, and repo-quality PRDs
- `PRD-DONE.md`, `PRD-fin.md`, and `INDEX.md`

Checked boxes are historical claims, not acceptance evidence. Current code, tests, packaged behavior, security boundaries, and user-visible results determine actual status.

## Source precedence

1. Unified Build Spec
2. Active Checklist
3. Team Lead Operating Rules
4. Current verified code and tests
5. Production-readiness and remaining-release-readiness trackers
6. Historical PRDs as feature and risk inputs

Conflicting historical requirements are retained only when they improve usefulness, intelligence, security, reliability, or usability and do not violate the higher-priority sources.

## Features to preserve: intelligence and orchestration

- Configurable multi-model routing by task type, model health, local/cloud preference, latency, privacy, and fallback availability.
- Automatic tool selection through a truthful capability registry rather than prompt-only guessing.
- LLM-assisted planning with bounded execution, dynamic replanning, retry/backoff, alternative paths, cancellation, and post-action verification.
- Execution-history learning and feedback summaries that improve future planning without silently changing security policy.
- Per-user preference learning with visible suggestions, editable defaults, and opt-out controls.
- Context assembly from identity, conversation state, memory, files, device/room state, task history, available tools, permissions, and response mode.
- Incoherent-output detection, current-information routing, uncertainty handling, and capability-limit recovery.
- Natural-language and script-based skill creation with validation, permissions, versioning, disable/delete controls, and safe invocation.

## Features to preserve: voice and ambient assistance

- Reliable Hey Rex wake word, Hold-to-Talk fallback, VAD, STT, TTS, barge-in, replay, device recovery, and measurable latency budgets.
- Voice enrollment and real-time speaker identification for James, Cole, and future authorized users.
- Voice selector, sample playback, personalized voice upload, and custom wake-word asset/training workflows.
- Room context, microphone/speaker origin, device aliases, fuzzy entity matching, and ambiguity clarification.
- Sonos/Bose or Home Assistant media endpoints for routed TTS, with local-network discovery and truthful health status.
- Voice-only, screen, hybrid, and automation response modes that never require an unseen display.
- Immediate acknowledgements, progressive status, model warmup, persistent STT/TTS processes, and acoustic thinking feedback where helpful.

## Features to preserve: desktop and mobile experience

- Electron remains the canonical desktop GUI and must expose every real capability plus its configurable settings, health, permissions, logs, and setup guidance.
- Guided first-run setup, truthful integration status, diagnostics, command history, quick actions, searchable logs, backup/restore, and safe update controls.
- Local file read/write, file search/summarization, program launching, Windows diagnostics, and policy-gated system actions.
- A usable Home Assistant dashboard with discovery, grouping, aliases, approval, control, undo where possible, and post-control state verification.
- Shared visual language across desktop and mobile: canonical logo assets, design tokens, typography, terminology, voice state, error states, and navigation patterns.
- Mobile chat, voice, history, tasks, reminders, notifications, settings, and mobile-native capabilities where appropriate.
- Secure mobile-to-desktop session interaction through device pairing and a capability broker; no blanket Desktop Commander access.
- Private mobile distribution with documented signing, secure updates, session recovery, and device revocation.

## Features to preserve: household data and integrations

- Per-user and household memory, selectable chat history, scoped document upload/vector indexing, and unified profile identity.
- Shared shopping list with voice/chat routing, desktop/mobile UI, ownership, and household sharing controls.
- Multiple email accounts per user, calendar, SMS, Telegram, notifications, quiet hours, digests, and escalation with truthful live/configured states.
- Home Assistant, Music Assistant, Plex, n8n, WordPress, WooCommerce, Nasteeshirts workflows, browser/file/code tooling, and business automations.
- OpenClaw/ClawHub dynamic skills and plugins as an optional external ecosystem with discovery, health, permissions, allow/deny lists, response normalization, retries, and verification.
- Phone integration remains a later optional workstream: inbound message taking/conversation and user-approved outbound calls, only after security and cost boundaries are explicit.
- Accurate time, date, timezone, location, and weather tools with user-configured defaults and privacy-conscious fallback behavior.

## Features to preserve: security, trust, and release quality

- Explicit identity on every request and strict per-user isolation for memory, email, calendar, history, shopping, credentials, tasks, and notifications.
- Per-user/per-device permissions, risk classification, confirmation gates, revocation, expiry, rate limits, replay protection, and tamper-evident audit records.
- Attempted, completed, verified, failed, and blocked are distinct action states throughout tools, voice, GUI, mobile, logs, and APIs.
- Secrets remain outside tracked configuration and are redacted from persisted settings, logs, exports, diagnostics, and error messages.
- Packaged Electron runtime, Python wheel, bridge resources, installer, managed runtime, mobile build, and update paths must be exercised by release gates.
- CI must enforce tests, lint, formatting, typechecks, dependency audits, secret scans, security audit, artifact cleanliness, working-tree cleanliness, and supported-surface smoke tests.
- Stub/mock behavior is test-only or visibly labeled; production paths fail closed with actionable errors.
- Windows 11 is the primary acceptance platform, with Windows 10 and macOS/Linux support classified honestly.

## Reconciliation decisions

### Retain as core or near-term

Voice reliability, model routing, context/memory, automatic tool selection, verified actions, desktop capability/settings parity, mobile pairing, Home Assistant, security hardening, observability, packaging, and release automation.

### Retain as later optional workstreams

Skill training, smart-speaker microphones, Telegram, Music Assistant, advanced autonomy learning, phone calls, and broad third-party business integrations. They remain in the product backlog and must not block a secure core release unless already presented as production-ready.

### Reject or supersede

- Any requirement that makes OpenClaw the mandatory Rex brain.
- Any PRD statement that mobile or multi-user support is a non-goal.
- Any architecture that treats the retired Flask/Tkinter surface as the canonical desktop app.
- Any acceptance claim based only on a checked checkbox, mock data, source-string tests, or unverified command dispatch.
- Any default that grants all paired devices arbitrary terminal, credential, file, or system-control access.

## Next use

This inventory is a preservation layer. The unified delivery backlog must map each retained capability to verified current state, implementation stories, tests, desktop/mobile surfaces, settings, permissions, documentation, and release evidence.