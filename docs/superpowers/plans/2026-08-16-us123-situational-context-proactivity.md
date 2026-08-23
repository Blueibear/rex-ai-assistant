# US-123 Situational Context and Proactive Assistance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one user-scoped situational-context/source-policy layer that preserves natural conversational references and produces high-signal proactive assistance while enforcing upload, disclosure, and location privacy as constitutional authority.

**Architecture:** Extend the existing `rex.context` builder/cache instead of creating another memory system. A canonical source-policy store decides whether data may enter broad context and who may receive it; bounded active references provide short-lived "it/that" continuity; a situational assembler reads only authorized sources; and a proactive evaluator ranks opportunities before handing at most one natural suggestion to the existing `SuggestionEngine`. Location and uploaded-document policy are enforced before retrieval, ranking, prompt construction, or proactive evaluation.

**Tech Stack:** Python 3.11, existing `rex.context`, `KnowledgeBase`, user/profile/permissions, calendar/current-info tools, SuggestionEngine, context cache revisions, Electron IPC/React, authenticated mobile settings, pytest/Vitest.

**Execution dependency:** Implement after US-121 and US-122 are merged or their commits are present in the execution branch; this plan consumes `rex.media`, output-routing policy, and the shared mobile settings route established by those plans.

## Global Constraints

- Connected-data availability, contextual eligibility, disclosure, and action authority are four separate decisions.
- Ordinary integrations deliberately connected by a user are context-eligible for that user by default unless disabled; contextual access never grants mutation authority.
- Every uploaded source independently records broad-context eligibility and audience scope; context-disabled files remain explicit-query-only.
- `location_assist` is explicit per user and `location_share` is a separate recipient-specific grant; household/admin status cannot override another user's choices.
- A denied location-disclosure request must not confirm whether current/recent location data exists.
- Rex, generated skills, OpenClaw capabilities, and developer/self-repair agents cannot autonomously widen these privacy/context grants.
- Revocation invalidates affected active references and context-cache revisions.
- Proactivity must be high-signal, explainable, user-scoped, dismissible, and unable to bypass normal action authorization/confirmation.
- No continuous location polling merely because permission exists.

---

### Task 1: Canonical context-source policy and cache revision contract

**Files:**
- Create: `rex/context/source_policy.py`
- Modify: `rex/context/revisions.py`
- Modify: `rex/context/builder.py`
- Test: `tests/context/test_source_policy.py`
- Modify/Test: `tests/rex2/test_context_cache_identity.py`

**Interfaces:**
- Produces: `ContextSourceType`, `AudienceScope`, `DisclosurePolicy`, `ContextSourcePolicy`, `ContextSourcePolicyStore`.
- `ContextSourcePolicyStore.is_context_eligible(source_id: str, *, subject_user_id: str, requester_user_id: str) -> bool` is called before source retrieval.
- `ContextSourcePolicyStore.revision_for_user(user_id: str) -> str` is content-free and feeds `ContextCacheVersions`.

- [x] **Step 1: Write failing eligibility/revision tests**

```python
def test_private_source_is_filtered_before_retrieval(tmp_path):
    store = ContextSourcePolicyStore(tmp_path)
    store.put(private_policy("upload:taxes", owner="james", context_enabled=True))
    assert store.is_context_eligible("upload:taxes", subject_user_id="james", requester_user_id="cole") is False


def test_policy_change_changes_content_free_revision(tmp_path):
    store = ContextSourcePolicyStore(tmp_path)
    before = store.revision_for_user("james")
    store.set_context_enabled("james", "calendar:main", False)
    assert store.revision_for_user("james") != before
```

- [x] **Step 2: Run focused tests and verify red state**
Run: `pytest -q tests/context/test_source_policy.py tests/rex2/test_context_cache_identity.py`
Expected: FAIL because canonical source policy does not exist.

- [x] **Step 3: Implement atomic policy storage and safe defaults**
Each record stores `source_id`, `source_type`, private owner if applicable, audience scope, `context_enabled`, disclosure policy, and monotonic/content-free revision metadata. Connected integrations registered by their owner default to `context_enabled=True`; uploads and location use their stricter task-specific defaults below.

- [x] **Step 4: Add policy revision to context cache versions**
`build_context_cache_versions()` hashes the policy revision, not raw policy/source content. `ContextBuilder` receives a source-policy service and never loads private source content before eligibility filtering.

- [x] **Step 5: Run context/cache tests and commit**
Run: `pytest -q tests/context/test_source_policy.py tests/rex2/test_context_cache.py tests/rex2/test_context_cache_identity.py tests/test_us014_context_builder.py`
Expected: PASS.
`git add rex/context tests/context/test_source_policy.py tests/rex2/test_context_cache_identity.py && git commit -m "feat(context): add canonical source policy"`

### Task 2: Upload policy, ownership, provenance, and safe legacy migration

**Files:**
- Modify: `rex/knowledge_base.py`
- Modify: `bridge/rex_memories_bridge.py`
- Modify: `rex/context/builder.py`
- Test: `tests/context/test_upload_policy.py`
- Modify/Test: `tests/test_knowledge_base.py`
- Modify/Test: `tests/test_us074_document_indexing.py`

**Interfaces:**
- Extend `KnowledgeDocument` with `owner_user_id: str | None`, `audience_scope`, `context_enabled: bool`, `disclosure_policy`, and `policy_revision: str`.
- Add `KnowledgeBase.search_for_user(query: str, *, requester_user_id: str, context_only: bool) -> list[KnowledgeDocument]`; filtering occurs before scoring/ranking.
- Explicit file operations may retrieve `context_enabled=False` documents only after normal file/document authorization.

- [x] **Step 1: Write failing cross-user/context-disabled/legacy tests**

```python
def test_context_disabled_upload_is_explicit_query_only(kb):
    kb.ingest_text("secret recipe", title="Recipe", owner_user_id="james", audience_scope="private", context_enabled=False)
    assert kb.search_for_user("recipe", requester_user_id="james", context_only=True) == []
    assert [d.title for d in kb.search_for_user("recipe", requester_user_id="james", context_only=False)] == ["Recipe"]


def test_legacy_unscoped_document_never_enters_background_context(kb_with_legacy_doc):
    assert kb_with_legacy_doc.search_for_user("legacy", requester_user_id="james", context_only=True) == []
```

- [x] **Step 2: Run upload tests and verify red state**
Run: `pytest -q tests/context/test_upload_policy.py tests/test_knowledge_base.py tests/test_us074_document_indexing.py`
Expected: FAIL because documents do not carry owner/context/audience policy.

- [x] **Step 3: Implement persisted document policy and migration**
New uploads require explicit private/household audience plus explicit context on/off. Existing unscoped documents migrate to a non-user-selectable `legacy_unassigned` state with `context_enabled=False`; they stay out of broad context until an authenticated owner assigns policy.

- [x] **Step 4: Filter before ranking and retain provenance**
Do not score/index-return a private document for an unauthorized requester. ContextBuilder includes citations/source IDs with derived facts so later summaries/proactive candidates retain enough provenance to recheck policy after revocation.

- [x] **Step 5: Run upload/KB regressions and commit**
Run: `pytest -q tests/context/test_upload_policy.py tests/test_knowledge_base.py tests/test_us074_document_indexing.py tests/test_memory_isolation.py`
Expected: PASS.
`git add rex/knowledge_base.py bridge/rex_memories_bridge.py rex/context/builder.py tests/context/test_upload_policy.py tests/test_knowledge_base.py tests/test_us074_document_indexing.py && git commit -m "feat(context): enforce upload context and audience policy"`

### Task 3: Location-assist and recipient-specific location-share authority

**Files:**
- Create: `rex/context/location_policy.py`
- Modify: `rex/geolocation.py`
- Modify: `rex/assistant.py`
- Modify: `rex/context/source_policy.py`
- Test: `tests/context/test_location_policy.py`
- Modify/Test: `tests/test_geolocation.py`

**Interfaces:**
- Produces: `LocationGrantStore`, `LocationContextService`, `LocationUsePurpose`.
- `LocationGrantStore.set_assist(*, owner_user_id: str, enabled: bool, actor_user_id: str)` requires `actor_user_id == owner_user_id`.
- `LocationGrantStore.set_share(*, owner_user_id: str, recipient_user_id: str, enabled: bool, actor_user_id: str)` is likewise owner-bound.
- `LocationContextService.get_for_assistance(user_id, purpose)` returns private location only when `location_assist` is enabled; `get_for_disclosure(subject_user_id, requester_user_id)` returns a generic denial when sharing is absent.

- [x] **Step 1: Write failing owner-only/admin-nonoverride/non-disclosure tests**

```python
def test_admin_cannot_enable_another_users_location(tmp_path):
    store = LocationGrantStore(tmp_path)
    with pytest.raises(PermissionError, match="owner authorization required"):
        store.set_assist(owner_user_id="cole", enabled=True, actor_user_id="james")


def test_denied_disclosure_does_not_confirm_location_presence(service):
    service.seed_private_location("cole", city="Dallas")
    result = service.get_for_disclosure(subject_user_id="cole", requester_user_id="james")
    assert result.message == "I can't share Cole's location."
    assert "Dallas" not in result.message
```

- [x] **Step 2: Run tests and verify red state**
Run: `pytest -q tests/context/test_location_policy.py tests/test_geolocation.py`
Expected: FAIL because current geolocation is process-global and has no per-user authority layer.

- [x] **Step 3: Put personal location behind `LocationContextService`**
Keep static configured household location/timezone separate from tracked/current user location. Existing IP-derived caching must never be interpreted as permission to track or disclose a user. Fetch current/recent personal location only for a materially relevant request/proactive rule and only after `location_assist`.

- [x] **Step 4: Remove ambient personal-location injection from Assistant**
Change `_build_tool_context` to accept the effective user and obtain any personal location through `LocationContextService`; denied/missing permission yields no personal location key. Explicit user-entered destination text remains usable without granting background tracking.

- [x] **Step 5: Run privacy/location regressions and commit**
Run: `pytest -q tests/context/test_location_policy.py tests/test_geolocation.py tests/test_assistant.py tests/test_memory_isolation.py`
Expected: PASS.
`git add rex/context/location_policy.py rex/geolocation.py rex/assistant.py rex/context/source_policy.py tests/context/test_location_policy.py tests/test_geolocation.py && git commit -m "feat(context): gate location assistance and sharing"`

### Task 4: Typed bounded active-context references and conversational resolution

**Files:**
- Create: `rex/context/active.py`
- Modify: `rex/context/builder.py`
- Modify: `rex/actions/dispatcher.py`
- Modify: `rex/media/sessions.py`
- Modify: `rex/timekeeping/tools.py`
- Test: `tests/context/test_active_context.py`
- Test: `tests/context/test_conversational_resolution.py`

**Interfaces:**
- Produces: `ActiveContextRef(domain, key, owner_user_id, payload, source_ids, revision, expires_at)`, `ActiveContextStore`, `ReferenceResolution`.
- `ActiveContextStore.put(ref)`, `get(user_id, domain, key)`, `resolve(user_id, utterance, candidate_domains)`, `invalidate_source(source_id)`.
- Domain adapters publish bounded IDs/state only; they never store whole prompts, transcripts, credentials, or private provider payloads.

- [x] **Step 1: Write failing expiry/ambiguity/cross-user tests**

```python
def test_it_resolves_to_recent_media_for_same_user(store):
    store.put(active_media_ref(user_id="james", key="session-1"))
    result = store.resolve("james", "pause it", candidate_domains=("media", "timekeeping"))
    assert result.ref.domain == "media"


def test_two_equally_relevant_refs_require_clarification(store):
    store.put(active_timer_ref("james", "timer-1"))
    store.put(active_timer_ref("james", "timer-2"))
    result = store.resolve("james", "cancel it", candidate_domains=("timekeeping",))
    assert result.ref is None
    assert result.reason == "ambiguous"
```

- [x] **Step 2: Run active-context tests and verify red state**
Run: `pytest -q tests/context/test_active_context.py tests/context/test_conversational_resolution.py`
Expected: FAIL because there is no canonical active-reference store.

- [x] **Step 3: Implement bounded references and source-revision invalidation**
Every read checks validated user ownership, expiry, and current source/policy revision. Revoked source policy clears matching refs immediately; stale refs are ignored rather than refreshed from unauthorized data.

- [x] **Step 4: Publish media/timekeeping references and expose them to TurnEngine routing**
US-121 media sessions publish `domain="media"`; timekeeping mutations/queries publish the exact record ID they just touched. Deterministic parsers may consume a resolved ref, while ContextBuilder may include a minimal active-state summary for LLM interpretation.

- [x] **Step 5: Run conversational regressions and commit**
Run: `pytest -q tests/context/test_active_context.py tests/context/test_conversational_resolution.py tests/media tests/timekeeping tests/test_assistant.py`
Expected: PASS.
`git add rex/context/active.py rex/context/builder.py rex/actions/dispatcher.py rex/media/sessions.py rex/timekeeping/tools.py tests/context && git commit -m "feat(context): add bounded conversational references"`

### Task 5: Situational assembler and high-signal proactive opportunity evaluator

**Files:**
- Create: `rex/context/situational.py`
- Create: `rex/proactivity/__init__.py`
- Create: `rex/proactivity/models.py`
- Create: `rex/proactivity/evaluator.py`
- Modify: `rex/suggestions/engine.py`
- Modify: `rex/response/builder.py`
- Modify: `rex/assistant.py`
- Test: `tests/proactivity/test_situational_assembler.py`
- Test: `tests/proactivity/test_evaluator.py`
- Modify/Test: `tests/test_us036_suggestions.py`

**Interfaces:**
- Produces: `SituationalSnapshot`, `ProactiveCandidate`, `ProactiveOpportunityEvaluator`.
- `ProactiveCandidate` fields: `key`, `user_id`, `spoken_text`, `source_ids`, `freshness_seconds`, `confidence`, `benefit`, `urgency`, `suggested_action`.
- Evaluator returns candidates sorted by deterministic score; `SuggestionEngine` still owns per-user pending/dismissal/session suppression.

- [x] **Step 1: Write failing cross-source and suppression tests**

```python
def test_commute_weather_candidate_combines_authorized_sources(evaluator):
    snapshot = commute_snapshot(calendar_destination="work", traffic_delay_minutes=18, storm_probability=0.8)
    candidate = evaluator.evaluate(snapshot)[0]
    assert candidate.key == "commute:weather-delay"
    assert "leave" in candidate.spoken_text.lower()


def test_private_upload_for_other_user_never_seeds_candidate(assembler):
    snapshot = assembler.build(user_id="cole")
    assert "upload:james-private" not in snapshot.source_ids
```

- [x] **Step 2: Run proactive tests and verify red state**
Run: `pytest -q tests/proactivity/test_situational_assembler.py tests/proactivity/test_evaluator.py tests/test_us036_suggestions.py`
Expected: FAIL because situational/proactive services do not exist.

- [x] **Step 3: Implement authorized snapshot assembly**
Source readers receive the current user and policy store. Initial readers: calendar events, relevant user memory/preferences, authorized contextual uploads, active capability/media/timekeeping state, and current-info adapters invoked only when a candidate rule materially needs fresh weather/traffic/search data. Preserve each fact's source ID/freshness.

- [x] **Step 4: Implement deterministic opportunity scoring and threshold**
Normalize `confidence`, `benefit`, and `urgency` to 0..1; compute `score = 0.45*benefit + 0.35*urgency + 0.20*confidence`; require `score >= 0.70` for ordinary conversational surfacing. Expired/stale critical inputs disqualify the candidate instead of lowering truth standards.

- [x] **Step 5: Reuse SuggestionEngine for delivery/dismissal rather than adding a second suggestion state machine**
Add `get_contextual_suggestion(candidates, *, user_id)` that applies existing one-per-session and dismissal rules. `ResponseBuilder` appends one natural “by the way” style suggestion only when the current response has not already asked a conflicting question; urgent out-of-turn delivery remains unavailable unless a separately authorized notification route exists.

- [x] **Step 6: Run proactivity/suggestion regressions and commit**
Run: `pytest -q tests/proactivity tests/test_us036_suggestions.py tests/test_suggestion_isolation.py tests/test_assistant.py`
Expected: PASS.
`git add rex/context/situational.py rex/proactivity rex/suggestions/engine.py rex/response/builder.py rex/assistant.py tests/proactivity tests/test_us036_suggestions.py && git commit -m "feat(context): add high-signal proactive assistance"`

### Task 6: Context/privacy Settings, mobile state, and constitutional mutation boundaries

**Files:**
- Create: `bridge/rex_context_policy_bridge.py`
- Create: `gui/src/main/handlers/contextPolicy.ts`
- Create: `gui/src/pages/settings/ContextPrivacySettingsSection.tsx`
- Modify: `gui/src/pages/SettingsPage.tsx`
- Modify: `gui/src/preload/index.ts`
- Modify: `gui/src/types/ipc.ts`
- Modify: `rex/mobile_api/routes/settings.py` from US-122
- Review: `rex/mobile_api/authorization.py` (existing `settings.read` / `settings.write` scopes are sufficient; privacy authority remains service-side)
- Test: `tests/context/test_policy_bridge.py`
- Test: `tests/mobile_api/test_context_privacy_settings.py`
- Test: `gui/tests/contextPrivacySettings.test.ts`

**Interfaces:**
- Operations: list/toggle connected contextual sources; inspect/change owned upload context/scope; set own `location_assist`; grant/revoke own location sharing for a named recipient; configure proactive-assistance preference.
- The service layer, not the renderer/mobile route, enforces owner-only privacy-authority mutations.

- [x] **Step 1: Write failing owner-bound bridge/mobile tests**

```python
def test_admin_cannot_change_cole_location_assist(client, james_admin_token):
    response = client.put("/mobile/settings/context/location", headers=james_admin_token, json={"user_id": "cole", "location_assist": True})
    assert response.status_code == 403


def test_uploader_can_promote_owned_upload_to_household(bridge):
    result = bridge({"action": "update_upload_policy", "user_id": "james", "doc_id": "doc_1", "audience_scope": "household", "context_enabled": True})
    assert result["ok"] is True
```

- [x] **Step 2: Run bridge/mobile/GUI tests and verify red state**
Run: `pytest -q tests/context/test_policy_bridge.py tests/mobile_api/test_context_privacy_settings.py`
Run: `cd gui && npm.cmd test -- --run tests/contextPrivacySettings.test.ts`
Expected: FAIL because the settings surface does not exist.

- [x] **Step 3: Implement service-first owner checks and safe renderer/mobile adapters**
The uploader/owner may change an owned document's contextual-use and audience. Only the tracked user may change their `location_assist` or recipient-specific `location_share`; admin permission is intentionally irrelevant to that mutation. Denials use generic messages that reveal no private current-location state.

- [x] **Step 4: Build Settings controls without technical leakage**
Use clear labels such as “Use this in future conversations,” “Private to me / Shared household,” “Use my location to help me,” and per-person “Share my location with …”. Show proactive-assistance controls separately from source/disclosure authority.

- [x] **Step 5: Add constitutional regression tests**
Exercise the same mutation service through direct Python, Electron bridge, mobile route, OpenClaw/developer-agent-shaped caller contexts, and assert no caller can self-widen privacy authority without the affected user/data-owner authorization.

- [x] **Step 6: Run Settings/privacy regressions and commit**
Run: `pytest -q tests/context/test_policy_bridge.py tests/context/test_location_policy.py tests/context/test_upload_policy.py tests/mobile_api/test_context_privacy_settings.py tests/test_memory_isolation.py`
Run: `cd gui && npm.cmd test -- --run tests/contextPrivacySettings.test.ts && npm.cmd run typecheck && npm.cmd run build`
Expected: PASS.
`git add CLAUDE.md bridge/README.md bridge/rex_context_policy_bridge.py gui/src gui/tests/contextPrivacySettings.test.ts rex/assistant.py rex/context/privacy.py rex/context/source_policy.py rex/mobile_api/routes/settings.py tests/context/test_policy_bridge.py tests/mobile_api/test_context_privacy_settings.py tests/proactivity/test_assistant_integration.py && git commit -m "feat(context): expose constitutional privacy controls"`

### Task 7: US-123 full validation, documentation, and release evidence

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `PRD-production-readiness.md`
- Modify: `docs/SELF_MAINTENANCE.md` only if implementation reveals a necessary clarification without weakening authority.
- Modify: `docs/superpowers/specs/2026-08-16-situational-context-media-privacy-design.md`
- Modify: `docs/planning/source-of-truth/REX_ACTIVE_CHECKLIST.md`

- [x] **Step 1: Run the complete privacy/context/proactivity focused matrix**
Run: `pytest -q tests/context tests/proactivity tests/media tests/output_routing tests/timekeeping tests/rex2/test_context_cache.py tests/rex2/test_context_cache_identity.py tests/test_knowledge_base.py tests/test_memory_isolation.py tests/test_suggestion_isolation.py tests/test_assistant.py tests/mobile_api/test_context_privacy_settings.py`
Expected: PASS.

- [x] **Step 2: Run GUI validation**
Run: `cd gui && npm.cmd test -- --run tests/contextPrivacySettings.test.ts tests/outputRoutingHandlers.test.ts && npm.cmd run typecheck && npm.cmd run build`
Expected: PASS.

- [x] **Step 3: Run static/security/repository integrity gates**
Run: `ruff check rex/context rex/proactivity rex/knowledge_base.py rex/geolocation.py rex/suggestions rex/assistant.py bridge/rex_context_policy_bridge.py tests/context tests/proactivity`
Run: `black --check rex/context rex/proactivity rex/knowledge_base.py rex/geolocation.py rex/suggestions rex/assistant.py bridge/rex_context_policy_bridge.py tests/context tests/proactivity`
Run: `mypy rex/context rex/proactivity rex/knowledge_base.py rex/geolocation.py rex/suggestions rex/assistant.py --ignore-missing-imports`
Run: `python scripts/security_audit.py --release-gate`
Run: `pytest -q tests/test_claude_truth.py tests/test_cross_doc_audit.py tests/test_repo_integrity.py tests/test_repository_integrity.py`
Run: `pre-commit run --all-files`
Run: `git diff --check`
Expected: all PASS.

- [x] **Step 4: Update tracker/docs with exact evidence and limitations**
Document which contextual sources have live adapters, that location is opt-in and non-disclosable without recipient-specific grant, that legacy unassigned uploads do not enter broad context, and that proactive traffic/weather accuracy depends on a configured current-info provider. Preserve Section 13 self-maintenance as post-RC and constitutional, not implemented behavior.

- [x] **Step 5: Run the repository release pytest matrix before PR freeze**
Run: `pytest -m "not slow and not audio and not gpu" -q`
Expected: zero failures. If an unrelated pre-existing environment hang/failure occurs, reproduce it from current master before classifying it as non-US-123 and document the evidence rather than claiming a full pass.
Result (2026-08-22): 9,325 passed, 65 skipped, 1 failed because the lean shared Python 3.11 environment does not have optional `openwakeword` installed. The exact failing wake-word reliability test fails identically on master; the three `master..origin/master` commits do not touch the test, loader, or dependency contract. This is inherited release-environment evidence, not a US-123 regression, so no full-pass claim is made.

- [x] **Step 6: Commit acceptance evidence**
`git add README.md CLAUDE.md PRD-production-readiness.md docs/SELF_MAINTENANCE.md docs/superpowers/specs/2026-08-16-situational-context-media-privacy-design.md docs/planning/source-of-truth/REX_ACTIVE_CHECKLIST.md && git commit -m "docs(context): record US-123 acceptance evidence"`

## US-123 Completion Check

Before the implementation PR is mergeable, prove with deterministic tests that: private uploads never influence another user's context; context-disabled uploads remain explicit-query-only; `location_assist` and recipient-specific `location_share` cannot be enabled by another user/admin; a denied location request reveals no presence/location data; policy revocation invalidates cached/active context; ambiguous active references clarify rather than guess; proactive suggestions are source-grounded, high-signal, user-scoped, and dismissal-aware; and no accepted suggestion bypasses normal mutation authorization.
