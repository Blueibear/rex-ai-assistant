# Rex 2.0 + Production Readiness Integration Design

**Date:** 2026-08-08
**Status:** Approved architecture amendment; implementation remains story-gated by `PRD-production-readiness.md`.

## Purpose

Integrate the reviewed Rex 2.0 intelligence/latency/tooling/memory/self-extension architecture into the remaining production-readiness work instead of finishing a transitional Rex and immediately refactoring it. This is an evolutionary migration, not a rewrite. Current production code remains authoritative until the story that replaces a boundary is implemented, tested, reviewed, and merged.

## Architectural destination

All user surfaces converge on one authenticated, identity-bound TurnEngine. A turn owns immutable user/device/scope context, cancellation, monotonic trace timing, context retrieval, model routing, capability selection, action planning, permission/risk checks, execution, verification, response events, and memory/experience recording. Streaming is a delivery mode of the same turn, never a less-capable intelligence path.

The event stream is the shared contract for CLI, Electron, voice, mobile, APIs, and future devices. Canonical events cover turn start/context/route/capability selection/plan/action progress/action evidence/model output/response sentences/terminal completion, failure, or cancellation. Exactly one terminal event is emitted.

## Capability architecture

One canonical Capability Registry replaces divergent capability/tool metadata authorities. A Capability/Tool Card records source, schema, operation type, permissions, health, risk, verification support, enabled state, and evidence. Local capabilities and OpenClaw/ClawHub capabilities adapt into the same model. Candidate retrieval is hybrid lexical + local semantic retrieval, but permission, health, risk, identity scope, and configuration filters run before ranking. Missing embeddings must degrade deterministically without a paid service.

OpenClaw remains optional. Rex remains the brain, identity authority, memory system, permission/risk policy, verifier, and final responder. Remote metadata can never widen Rex authority.

## Action and verification architecture

The canonical action lifecycle is: `planned -> authorized -> attempted -> completed -> verified`, with terminal/branch states `unverified`, `failed`, and `cancelled`. Invalid transitions fail closed. An exception-free tool return is not verification. After cancellation or transport loss following a mutation, unknown real-world outcome is `attempted/unverified`, never fabricated as failure or success.

A minimal dependency graph allows only independent, permitted nodes to run in parallel. Conflicting or mutating operations serialize around explicit dependencies and confirmation/commit boundaries.

## Latency architecture

Optimization begins with an accurate RexBench baseline, separated by request class and evidence class. Then the runtime can safely add managed warm local components, streaming ASR plus semantic endpointing, bounded speculative read-only prefetch, sentence/clause TTS, barge-in via canonical cancellation, progressive event-derived status, and identity-safe prompt/context caches. Optimizations stay inside security boundaries.

## Model architecture

ModelRouter 2.0 separates fast/local executive decisions from deep reasoning. Complexity and confidence are explicit routing evidence. Low confidence may escalate once through configured providers; no cloud or paid provider is silently enabled. Provider health/reliability/cooldown data is privacy-safe and evaluation-backed.

## Memory architecture

Memory declares type (`semantic`, `episodic`, `preference`) and scope (`private user`, `household`) before ranking. James/Cole isolation applies to retrieval, caches, events, tools, cancellation, and procedural learning. Procedural experience is separate from normal memory: only verified outcomes become candidates, mutations require human approval, and procedures carry provenance, versions, success/failure counts, expiry/revalidation, revocation, and risk.

## Capability gaps and Forge

When Rex cannot perform a task, it searches in order: enabled local capabilities, disabled local capabilities, OpenClaw/ClawHub, configured MCP providers, configured OpenAPI descriptions, and safe declarative composition. It does not jump directly to generated code.

Forge comes last. Declarative composition is preferred. Generated capability packages require a manifest, bounded permissions, tests, sandboxed build/test, static/security analysis, RexBench/adversarial evaluation, risk-based approval, canary deployment, and atomic rollback/revocation. Generated code receives no execution authority merely because it was generated, and may never receive more authority than the capabilities used to build/test it without explicit human approval. Initial autonomous promotion is limited to read-only low-risk capabilities.

## Security invariants

- Explicit validated identity remains fail-closed; no legacy/default fallback may silently cross user boundaries.
- James/Cole private state, prompts, events, caches, memory, tools, cancellation, and experience remain isolated.
- Existing desktop-owned mobile pairing, live grants, revocation, TLS binding, strong authentication, rate limits, and least-privilege scopes remain authoritative.
- SMS remains a supported backend/direct route but is intentionally absent from primary navigation.
- No new paid service is required by this architecture; local-first operation remains mandatory.
- OpenClaw absence/outage cannot remove core Rex functionality.
- Verification evidence, not optimistic wording, controls user-facing success claims.

## Evidence classes

Benchmarks and release claims must label evidence as one of: deterministic/mock, local source runtime, live provider, packaged Windows artifact, or physical hardware/device. Mock performance must never be presented as physical-device or live-provider performance.

## Migration order rationale

The explicit execution order in `PRD-production-readiness.md` starts with the current Settings decomposition, establishes a pre-change latency baseline and capability inventory, then unifies the TurnEngine before further response/voice/mobile work. Capability/action foundations follow before dynamic OpenClaw and gap recovery. Voice latency work then lands on the canonical runtime rather than a transitional path. Identity/memory/mobile work follows the shared event contract. Forge and final RexBench are deliberately last. This minimizes duplicated implementation while keeping every intermediate commit reviewable and functional.
