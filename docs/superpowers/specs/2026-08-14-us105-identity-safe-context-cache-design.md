# US-105 Identity-Safe Context Cache Design

## Purpose

US-105 reduces repeated prompt/context assembly work without allowing cached private context to cross identity, scope, policy, model, configuration, capability, or memory boundaries.

The safety rule is fail-closed: a cache miss is always preferable to a stale or wrong-user hit. The cache is an optimization only; disabling or bypassing it must preserve assistant behavior.

## Chosen approach

Cache only deterministic context *artifacts*, not complete prompts, chat histories, tool results, follow-up state, user messages, or final `ContextPackage` objects.

The primary cached private artifact contains immutable copies of:
- active personality prompt,
- formatted user profile context,
- formatted remembered-facts context,
- user-facts key/value pairs.

Dynamic system date/time, current conversation history, current user message, tool context, follow-up cues, response mode, and action results remain uncached and are assembled fresh each turn.
## Cache key contract

A private cache key requires a validated `user_id` and `TurnScope.USER`. A household cache key is possible only through an explicit `TurnScope.HOUSEHOLD` request and carries no private user owner.

Every key also contains content-free revision tokens for:
- identity,
- policy and permission snapshot,
- selected model,
- capability registry metadata/runtime state,
- relevant non-secret context configuration,
- user memory/profile/facts,
- prompt-template schema.

Revision tokens are SHA-256 digests over deterministic, bounded metadata. Raw private content, credentials, prompts, transcripts, tool payloads, and user facts never appear in keys or metrics.

A key change is deterministic invalidation: stale entries may remain physically until bounded LRU eviction, but they become unreachable immediately.
## Revision sources

`rex.context.revisions` owns deterministic revision helpers:
- model revision from selected provider/model identity,
- policy/permission revision from the immutable `AuthorizationSnapshotRef`,
- capability revision from the canonical `CapabilityRegistry.metadata_snapshot()`,
- config revision from only context-relevant, non-secret settings,
- identity/memory revision from the validated user's profile/facts files and their contents,
- prompt-template revision from an explicit constant that changes when context injection semantics change.

Private file revisions hash file contents with bounded reads and explicit missing-file markers. This avoids relying on timestamp granularity while keeping raw content out of the key.

The canonical `Assistant` passes the current immutable turn scope and authorization snapshot plus selected-model information into `ContextBuilder`. Legacy/direct `ContextBuilder.build()` callers that do not supply a complete safe cache request bypass cross-turn caching.

## Concurrency and bounds

`ContextArtifactCache` is thread-safe and size-bounded. Builders execute outside the bookkeeping lock; duplicate same-key builds are acceptable because correctness is preserved and long global lock holds are avoided.

Cached artifact values are immutable so one caller cannot mutate data later observed by another request.
## Metrics and logging

Metrics expose only bounded categories and timings: hit count, miss count, build count, eviction count, total build seconds, and current entry count. They never include user IDs, cache-key digests, prompt text, memory content, credentials, filenames, or tool payloads.

Normal cache misses are silent/DEBUG-level operational events. Cache construction failures fail open to uncached context assembly and do not make Rex unavailable.

## Alternatives rejected

1. **Cache full prompts or `ContextPackage` objects.** Rejected because history, messages, follow-ups, tool context, and current user input are dynamic/private and would produce both low-value hits and much larger privacy risk.
2. **Use TTL-only invalidation.** Rejected because policy/model/config/memory changes must invalidate immediately and deterministically.
3. **Share all non-message context at household scope.** Rejected because personality, preferences, facts, and profile context are private even when two users share a device.

## Verification

Tests must prove:
- private James/Cole keys can never collide,
- household reuse occurs only with explicit household scope,
- each required revision change forces a miss,
- concurrent James/Cole builds never return the other user's artifact,
- metrics contain no private content,
- ContextBuilder output remains behaviorally equivalent with cache hits and misses.

No new dependency is required.