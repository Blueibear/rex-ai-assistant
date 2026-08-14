# US-105 Identity-Safe Context Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse deterministic prompt/context artifacts without ever returning stale or cross-user private context.

**Architecture:** Add a small bounded `ContextArtifactCache` plus deterministic revision helpers under `rex/context/`. `ContextBuilder` caches only immutable personality/profile/facts artifacts when the canonical Assistant supplies a complete identity/scope/authorization/model cache request; dynamic time, history, user messages, tool context, follow-ups, and final prompts remain freshly assembled every turn.

**Tech Stack:** Python 3.11, frozen dataclasses, `threading.Lock`, `hashlib.sha256`, deterministic JSON, pytest.

## Global Constraints

- Private cache entries require a validated user ID and `TurnScope.USER`; ambiguous identity bypasses caching.
- Household reuse is allowed only through explicit `TurnScope.HOUSEHOLD` and must not carry private user artifacts.
- Keys/invalidation include identity, scope, policy, permission, model, capability, non-secret config, memory, and prompt-template revisions.
- Metrics/logging must never contain raw prompts, transcripts, memory/facts, credentials, filenames, user IDs, or tool payloads.
- Cache failure must degrade to uncached context assembly, not assistant failure.
- No new dependency and no change to core Rex behavior when caching is unavailable.

---
### Task 1: Core cache contracts and privacy-safe metrics

**Files:**
- Create: `rex/context/cache.py`
- Create: `tests/rex2/test_context_cache.py`

**Interfaces:**
- Produces `ContextCacheVersions`, `ContextCacheKey`, `ContextArtifactCache[T]`, and `ContextCacheMetrics`.
- `ContextCacheKey.private(user_id, versions)` validates the user and binds `TurnScope.USER`.
- `ContextCacheKey.household(versions)` binds `TurnScope.HOUSEHOLD` with no owner.
- `ContextArtifactCache.get_or_build(key, builder) -> T` builds outside the global bookkeeping lock.

- [x] **Step 1: Write failing cache-contract tests**

```python
def test_private_key_requires_validated_user_and_differs_by_owner():
    versions = _versions()
    james = ContextCacheKey.private("james", versions)
    cole = ContextCacheKey.private("cole", versions)
    assert james != cole
    with pytest.raises(ValueError):
        ContextCacheKey.private("../james", versions)
```

Also assert explicit household keys have `user_id is None`, LRU size stays bounded, repeated keys call the builder once, and metrics expose only numeric category counters/timing.

- [x] **Step 2: Run `pytest tests/rex2/test_context_cache.py -q` and verify RED because `rex.context.cache` does not exist.**
- [x] **Step 3: Implement the minimal frozen key/version/metrics dataclasses and bounded thread-safe cache.**
- [x] **Step 4: Re-run `pytest tests/rex2/test_context_cache.py -q` and verify GREEN.**
### Task 2: Deterministic revision snapshot and invalidation

**Files:**
- Create: `rex/context/revisions.py`
- Modify: `tests/rex2/test_context_cache.py`

**Interfaces:**
- Produces frozen `ContextCacheRequest(user_id, scope, authorization, model_provider, model_name)`.
- Produces `build_context_cache_versions(request, settings, capability_registry=None) -> ContextCacheVersions`.
- Uses `CONTEXT_PROMPT_TEMPLATE_REVISION` as an explicit schema/version boundary.

- [ ] **Step 1: Add failing revision tests.**

```python
@pytest.mark.parametrize("field", ["policy", "permission", "model", "capability", "config", "memory"])
def test_each_relevant_revision_change_changes_the_cache_versions(field, revision_fixture):
    before, after = revision_fixture(field)
    assert before != after
```

Tests must additionally prove raw James/Cole facts, API-token-shaped settings, filenames, and profile text do not occur in the returned version tokens or their serialized representation.

- [ ] **Step 2: Run the new revision tests and verify RED because revision helpers are missing.**
- [ ] **Step 3: Implement deterministic SHA-256 revision helpers over bounded metadata/content and explicit missing-file markers.**
- [ ] **Step 4: Fingerprint only context-relevant non-secret config, canonical capability metadata/runtime state, authorization refs, selected provider/model, and validated identity/memory files.**
- [ ] **Step 5: Re-run `pytest tests/rex2/test_context_cache.py -q` and verify GREEN.**
### Task 3: Cross-user and household isolation under concurrency

**Files:**
- Create: `tests/rex2/test_context_cache_identity.py`
- Modify: `rex/context/cache.py` only if the failing tests require it.

**Interfaces:**
- Consumes the cache/key/version interfaces from Tasks 1-2.
- Proves isolation independently of `ContextBuilder` so the security contract is directly testable.

- [ ] **Step 1: Write failing identity/concurrency tests.**

```python
def test_concurrent_private_builds_never_cross_users():
    # Run James and Cole on one cache with a barrier inside the builders.
    # Each returned artifact must contain only its own marker.
    assert results["james"] == "artifact-for-james"
    assert results["cole"] == "artifact-for-cole"
```

Add tests that household reuse occurs only through `ContextCacheKey.household(...)`, and that a private key can never equal a household key even with identical revisions.

- [ ] **Step 2: Run `pytest tests/rex2/test_context_cache_identity.py -q` and verify RED for any missing isolation behavior.**
- [ ] **Step 3: Make only the minimal cache changes required for safe concurrent behavior; do not serialize builders under the global cache lock.**
- [ ] **Step 4: Re-run both context-cache test files and verify GREEN.**
### Task 4: ContextBuilder artifact caching without behavior drift

**Files:**
- Modify: `rex/context/builder.py`
- Modify: `tests/test_us014_context_builder.py`
- Modify: `tests/rex2/test_context_cache.py`

**Interfaces:**
- Add frozen `PrivateContextArtifacts` containing personality, profile context, facts context, and immutable user-fact pairs.
- Extend `ContextBuilder.build(..., cache_request: ContextCacheRequest | None = None)`.
- A complete safe request may use the cache; missing/mismatched request data bypasses it.

- [ ] **Step 1: Write failing builder tests proving one artifact load serves both message and text-prompt assembly and a repeated safe request reuses it.**
- [ ] **Step 2: Add a failing equivalence test comparing cached-hit output with uncached output while history, current message, tool context, and voice mode change normally.**
- [ ] **Step 3: Run the focused tests and verify RED because `cache_request`/artifact caching do not exist.**
- [ ] **Step 4: Implement `_build_private_artifacts()` once per build, immutable cached artifacts, fresh dict copies for `ContextPackage.user_facts`, and cache-failure fallback to uncached assembly.**
- [ ] **Step 5: Re-run `pytest tests/test_us014_context_builder.py tests/rex2/test_context_cache.py -q` and verify GREEN.**
### Task 5: Wire canonical Assistant turns into the safe cache boundary

**Files:**
- Modify: `rex/assistant.py`
- Modify: `tests/rex2/test_generate_reply_turn_engine.py`

**Interfaces:**
- Build `ContextCacheRequest` from the current `TurnContext` plus current `self._llm.provider` and `self._llm.model_name`.
- Pass the request only to the canonical `ContextBuilder.build()` call inside `_run_reply_turn`.

- [ ] **Step 1: Add a failing turn-engine test that captures `context_builder.build.call_args.kwargs["cache_request"]`.**

```python
request = context_builder.build.call_args.kwargs["cache_request"]
assert request.user_id == "james"
assert request.scope is TurnScope.USER
assert request.authorization == observed_turn_context.authorization
assert request.model_name == "test-model"
```

- [ ] **Step 2: Run that test and verify RED because no cache request is currently supplied.**
- [ ] **Step 3: Implement the minimal canonical wiring without changing identity resolution or mutating Assistant user state.**
- [ ] **Step 4: Run `pytest tests/rex2/test_generate_reply_turn_engine.py tests/rex2/test_context_cache*.py tests/test_us014_context_builder.py -q` and verify GREEN.**
### Task 6: Documentation, tracker reconciliation, and release verification

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/performance.md`
- Modify: `PRD-production-readiness.md`
- Modify: `docs/archive/progress/progress-production-readiness.txt`

**Interfaces:**
- Document context-cache privacy/invalidation rules and content-free metrics.
- Reconcile US-099's final GitHub checkbox with PR #395 exact-head evidence before updating US-105.

- [ ] **Step 1: Update docs and tracker only after implementation tests are green. Leave US-105's GitHub-check criterion unchecked until CI passes on the PR head.**
- [ ] **Step 2: Run `pytest tests/rex2/test_context_cache.py tests/rex2/test_context_cache_identity.py tests/rex2/test_generate_reply_turn_engine.py tests/test_us014_context_builder.py -q`.**
- [ ] **Step 3: Run Ruff, Black, and MyPy on all modified Python files plus `git diff --check`.**
- [ ] **Step 4: Run `python scripts/security_audit.py --release-gate` and verify no private content/secrets are exposed.**
- [ ] **Step 5: Run the primary CI marker set `pytest -m "not slow and not audio and not gpu" -q` before publishing.**
- [ ] **Step 6: Commit using Conventional Commits, rebase onto current `origin/master`, rerun conflict-sensitive tests, push, open a PR, and verify all required GitHub workflows on the exact head before squash-merge.**

## Plan self-review

- All US-105 acceptance criteria map to Tasks 1-6.
- No full prompt/history/tool result is cached.
- USER/HOUSEHOLD scope behavior is explicit and fail-closed.
- Every production behavior change starts with a failing test.
- No new dependency, secret-bearing telemetry, or implicit default identity is introduced.