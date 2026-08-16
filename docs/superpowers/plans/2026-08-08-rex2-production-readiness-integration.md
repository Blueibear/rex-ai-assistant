# Rex 2.0 + Production Readiness Implementation Plan

**Date:** 2026-08-08
**Authority:** `PRD-production-readiness.md` is the detailed story tracker; this plan groups the work and names implementation boundaries. The mandatory timers/alarms/speaker-routing addendum is `docs/superpowers/specs/2026-08-15-timers-alarms-media-routing.md`.

## Phase A ? Finish current structural work and measure baseline

- **US-063:** finish Settings section decomposition and packaged Settings smoke.
- **US-075:** create `scripts/rexbench.py` plus `tests/rex2/` deterministic request-class fixtures; record cold/warm p50/p95 and stage timings without sensitive payloads.
- **US-064:** inventory capability authorities/consumers/metadata and establish the migration map for the canonical registry.

Checkpoint: current behavior is measurable before architectural migration.

## Phase B ? Canonical Turn Runtime

- **US-094:** add `rex/runtime/turn.py`, `events.py`, `turn_engine.py` (initial facade), and contract tests under `tests/rex2/test_turn_contracts.py`.
- **US-095:** route `Assistant.generate_reply()` through TurnEngine while preserving public behavior.
- **US-096:** route `Assistant.stream_reply()` through the same engine and prove parity.
- **US-076:** move output validation to canonical turn completion and validate escalated outputs.
- **US-097:** migrate CLI/Electron/voice/mobile/API adapters so none bypass TurnEngine.
- **US-098:** add idempotent turn cancellation through model/tool/OpenClaw/TTS paths and truthful post-mutation unknown-state semantics.

Checkpoint: one brain, one event model, one terminal outcome contract across surfaces.

## Phase C ? Capability and action intelligence

- **US-106:** consolidate `rex/capabilities` / tool metadata into one Capability Registry and adapters.
- **US-107:** implement permission-aware hybrid lexical + local semantic candidate retrieval.
- **US-109:** generalize the canonical action lifecycle and immutable evidence correlation.
- **US-108:** implement bounded dependency graphs and safe parallel execution.
- **US-104:** expose progressive status only from canonical events.

Checkpoint: tool selection/execution is inspectable, permission-aware, parallel where safe, and verification-first.

## Phase D ? Latency and model routing

- **US-099:** managed warm lifecycle for executive/STT/TTS/indexes.
- **US-105:** user/policy/model-version-safe prompt/context caches.
- **US-110:** ModelRouter 2.0 fast/deep routing and one-step confidence escalation.
- **US-111:** provider reliability/cooldown metrics and deterministic routing eval corpus.
- **US-071/072/073/077:** finish provider persistence/discovery/autonomy/current-information UX on those canonical services.
- **US-078:** capability-gap recovery consults the canonical registry/ecosystems before any build suggestion.
- **US-113/114:** dynamic OpenClaw discovery/hot refresh/reconnect/verification hardening.
- **US-101:** bounded speculative read-only prefetch after capability/action safety exists.

Checkpoint: Rex is faster without reducing intelligence or bypassing security.

## Phase E ? Voice on the canonical runtime

- **US-074/068/069/070:** finish diagnostics, identity enrollment, wake assets, and Test Voice surface.
- **US-100:** streaming ASR with semantic endpointing and correction-before-dispatch.
- **US-102:** ordered sentence/clause TTS from turn events.
- **US-103:** barge-in via turn cancellation.
- **US-079/067:** typed-chat speak preference and timezone UI on the shared response/context path.

Checkpoint: fast voice is cancellable, evidence-aware, and uses the same intelligence path as text.

## Phase F ? Remaining GUI/integrations, identity, memory, mobile, household audio

- **US-065/066/080/081/082:** truthful GUI integration/navigation/HA/Outlook/Email-SMS work.
- **US-087:** cross-surface identity invariants and James/Cole concurrency tests.
- **US-083/084/085/086:** history, shopping, typed/scoped semantic memory, and uploads/vector indexing. US-086 additionally separates per-file context inclusion from private/household audience scope and preserves source provenance.
- **US-112:** guarded procedural experience memory.
- **US-088:** mobile chat/voice consumes TurnEngine events while preserving existing secure pairing/strong-auth/TLS/revocation boundaries.
- **US-120:** implement first-class concurrent timers and alarms with naming, recurrence, snooze/dismiss, restart recovery, per-user ownership, and canonical tool exposure.
- **US-121:** implement canonical speaker/room/group discovery plus provider-neutral media targets/accounts, request-origin playback defaulting, active media-session context, persistent speaker groups, and verified playback controls.
- **US-122:** add per-user output/media-account routing policies and Settings UI for spoken responses, timers, alarms, and media, including explicit target overrides, request-origin behavior, time-of-day rules, quiet hours, target volume, and unavailable-target fallback behavior.
- **US-123:** add canonical situational-context/source policy and proactive opportunity evaluation, including explicit per-user location assist, separate recipient-specific location sharing, generalized active references, and cross-user privacy isolation.

Detailed acceptance criteria for US-120 through US-122 remain in `docs/superpowers/specs/2026-08-15-timers-alarms-media-routing.md`; the refined US-086/121/122 requirements and US-123 design are in `docs/superpowers/specs/2026-08-16-situational-context-media-privacy-design.md`.

Checkpoint: every user-facing surface shares identity, turn, memory, verification, timer/alarm semantics, household audio-routing semantics, and one privacy-aware contextual-source policy.

## Phase G ? Safe self-extension and release gate

- **US-115:** declarative capability composition first.
- **US-116:** Forge package/manifest/sandbox/tests/security/RexBench pipeline.
- **US-117:** approval, low-risk-only initial auto-promotion, canary, rollback/revocation.
- **US-089?093:** retire justified compatibility/generated skips.
- **US-118:** final production RexBench across performance, privacy, escalation, failures/outages, Forge adversarial cases, Windows Electron, mobile, physical voice evidence, timer/alarm timing and recovery, speaker-group/media-account routing, request-origin behavior, uploaded-context isolation, location permission/non-disclosure, proactive-context behavior, per-user routing isolation, and unavailable-target behavior.

Checkpoint: release candidate evidence is explicit and no mock result is mislabeled as live/hardware proof. US-120, US-121, US-122, and US-123 must be complete before this release gate can pass.

## Per-story working method

For each story: read `CLAUDE.md`; write failing focused tests first where behavior changes; implement only the story scope; run targeted regressions; run required Ruff/Black/mypy/GUI gates; update `CLAUDE.md` when architecture/commands/config/dependencies/integrations change; update the authoritative PRD and progress ledger in the same commit; push one story PR; verify GitHub checks independently; close the GitHub criterion only with evidence; then merge under the standing user authorization.
