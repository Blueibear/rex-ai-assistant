# US-122 Per-User Output Routing Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each user one canonical policy for media accounts and spoken/timer/alarm/media output targets, with request-origin defaults, explicit overrides, conditions, safe fallback, and shared desktop/mobile settings state.

**Architecture:** Add `rex.output_routing` as policy only, above US-121's target/account registry and below voice/timekeeping/media execution. Policies are persisted per user; a small household record may identify the ordinary-playback primary media account but never grants library-mutation authority. Resolution returns a typed decision explaining target/account/fallback/volume, and callers retain normal permission/action authority.

**Tech Stack:** Python 3.11, Pydantic/dataclasses, existing user data paths/profile service/credential vault, `rex.media`, `rex.timekeeping`, TurnInvocation/TurnContext, Electron IPC/React, Flask mobile API, pytest/Vitest.

**Execution dependency:** Implement after the US-121 plan is merged or its commits are present in the execution branch; this plan intentionally modifies `rex.media` interfaces created by US-121.

## Global Constraints

- Output/account routing is policy, never authority; it cannot widen device permissions or borrow another user's private library mutation authority.
- Explicit natural-language target/account selection outranks stored defaults when currently authorized.
- Interactive media prefers the authorized trusted request-origin endpoint when no explicit target is named.
- High-confidence voice identity uses that user's linked/default media account; unresolved identity may use a configured household primary account only for ordinary playback.
- Quiet hours/fallback must never silently suppress explicitly required or safety-relevant delivery.
- Electron and mobile/PWA must read/write the same backend policy service, not duplicate local settings.
- Per-user privacy/context rules from US-123 remain independent of output routing.

---

### Task 1: Routing policy models, persistence, and deterministic resolver

**Files:**
- Create: `rex/output_routing/__init__.py`
- Create: `rex/output_routing/models.py`
- Create: `rex/output_routing/service.py`
- Test: `tests/output_routing/test_policy_service.py`

**Interfaces:**
- Produces: `OutputKind`, `FallbackMode`, `RoutingRule`, `UserOutputPolicy`, `ResolvedRoute`, `OutputRoutingService`.
- `OutputRoutingService.resolve(*, user_id: str, output_kind: OutputKind, explicit_target_text: str | None, origin_device_id: str | None, at: datetime) -> ResolvedRoute` consumes US-121 `AudioTargetRegistry`.

- [ ] **Step 1: Write failing precedence/condition/fallback tests**

```python
def test_explicit_target_beats_origin_and_default():
    route = service.resolve(user_id="james", output_kind=OutputKind.MEDIA, explicit_target_text="living room", origin_device_id="mic_kitchen", at=NOON)
    assert route.target_id == "ha:media_player.living_room"
    assert route.reason == "explicit_target"


def test_no_fallback_does_not_silently_reroute():
    service.save_policy("james", UserOutputPolicy(media_target_id="offline:x", media_fallback=FallbackMode.NONE))
    route = service.resolve(user_id="james", output_kind=OutputKind.MEDIA, explicit_target_text=None, origin_device_id=None, at=NOON)
    assert route.target_id is None
    assert route.reason == "configured_target_unavailable"
```

- [ ] **Step 2: Run focused test and verify red state**
Run: `pytest -q tests/output_routing/test_policy_service.py`
Expected: FAIL because routing policy types/service do not exist.

- [ ] **Step 3: Implement atomic per-user persistence and resolver precedence**
Order: explicit target -> interactive media origin target -> matching time/day rule -> per-kind default -> configured fallback. Persist policy under canonical user data, validate every target ID against US-121 registry at resolution time, and return reason/fallback metadata without private content.

- [ ] **Step 4: Add quiet-hours and temporary target-volume semantics**
`ResolvedRoute.target_volume` is advisory to delivery code; do not mutate a device's permanent normal volume unless policy explicitly says to persist it. Quiet hours return a structured suppression/alternate decision instead of silently dropping delivery.

- [ ] **Step 5: Run tests to green and commit**
Run: `pytest -q tests/output_routing/test_policy_service.py`
Expected: PASS.
`git add rex/output_routing tests/output_routing/test_policy_service.py && git commit -m "feat(routing): add per-user output policy"`

### Task 2: Trusted speaker-identity provenance and media-account selection

**Files:**
- Modify: `rex/runtime/invocation.py`
- Modify: `rex/runtime/turn.py`
- Modify: `rex/assistant.py`
- Modify: `rex/voice_identity/fallback_flow.py`
- Modify: `rex/media/accounts.py`
- Modify: `rex/output_routing/service.py`
- Test: `tests/output_routing/test_account_resolution.py`
- Modify/Test: `tests/test_voice_identity_fallback.py`

**Interfaces:**
- Add trusted `IdentityResolution` values `explicit`, `voice_recognized`, `voice_review`, `fallback`, `unknown` to invocation/turn context; renderer/transcript content cannot set this authority field.
- `OutputRoutingService.resolve_media_account(*, active_user_id: str, identity_resolution: IdentityResolution, requested_account_id: str | None, operation: str) -> MediaAccountRef | None`.

- [ ] **Step 1: Write failing account-selection tests**

```python
def test_recognized_speaker_uses_own_account():
    account = service.resolve_media_account(active_user_id="cole", identity_resolution=IdentityResolution.VOICE_RECOGNIZED, requested_account_id=None, operation="play")
    assert account.owner_user_id == "cole"


def test_unknown_speaker_primary_account_cannot_mutate_library():
    with pytest.raises(PermissionError, match="identity required"):
        service.resolve_media_account(active_user_id="james", identity_resolution=IdentityResolution.UNKNOWN, requested_account_id=None, operation="favorite")
```

- [ ] **Step 2: Run focused tests and verify red state**
Run: `pytest -q tests/output_routing/test_account_resolution.py tests/test_voice_identity_fallback.py`
Expected: FAIL on missing trusted identity-resolution metadata/account resolver.

- [ ] **Step 3: Stamp identity resolution inside trusted voice/runtime adapters**
Recognized voice -> `voice_recognized`; review accepted because session agrees -> `voice_review`; identity-chain fallback -> `fallback`; no identity evidence -> `unknown`; typed/profile-selected turns -> `explicit`. `Assistant._build_turn_context()` copies this trusted invocation metadata.

- [ ] **Step 4: Implement account precedence and primary-account boundary**
Explicit authorized account -> recognized user's default -> household primary only for non-library-mutating ordinary playback when identity is unresolved. Never return another user's private account for favorite/playlist/library/profile mutations.

- [ ] **Step 5: Run identity/account regressions and commit**
Run: `pytest -q tests/output_routing/test_account_resolution.py tests/test_voice_identity_fallback.py tests/rex2/test_turn_contracts.py tests/test_user_profile_service.py`
Expected: PASS.
`git add rex/runtime rex/assistant.py rex/voice_identity/fallback_flow.py rex/media/accounts.py rex/output_routing/service.py tests/output_routing/test_account_resolution.py tests/test_voice_identity_fallback.py && git commit -m "feat(routing): bind media accounts to trusted identity"`

### Task 3: Timer/alarm target parsing and due-event delivery

**Files:**
- Modify: `rex/timekeeping/parser.py`
- Modify: `rex/timekeeping/tools.py`
- Modify: `rex/timekeeping/runtime.py`
- Modify: `rex/timekeeping/models.py` only if route metadata beyond existing `output_target_id` is required.
- Create: `rex/output_routing/delivery.py`
- Test: `tests/output_routing/test_timekeeping_delivery.py`
- Modify/Test: `tests/timekeeping/test_parser.py`
- Modify/Test: `tests/timekeeping/test_tools.py`

**Interfaces:**
- Extend `TimekeepingCommand` with `target_text: str | None` and optional `target_volume: int | None`; persisted records continue to store canonical `output_target_id`.
- `OutputDeliveryService.deliver_due_event(event: DueEvent) -> DeliveryResult` resolves current policy/fallback at fire time while honoring an event's explicit target first.

- [ ] **Step 1: Write failing explicit-target and outage-fallback tests**

```python
def test_alarm_explicit_target_is_persisted():
    command = parse_timekeeping_command("set an alarm for 7 and play it on the bedroom speaker", user_timezone="America/Chicago")
    result = timekeeping_manage(command=command, user_id="james")
    assert result["record"]["output_target_id"] == "ha:media_player.bedroom"


def test_due_event_uses_current_fallback_if_default_is_offline():
    result = delivery.deliver_due_event(due_event(target_id=None, owner="james"))
    assert result.target_id == "ha:media_player.kitchen"
    assert result.reason == "named_fallback"
```

- [ ] **Step 2: Run focused tests and verify red state**
Run: `pytest -q tests/output_routing/test_timekeeping_delivery.py tests/timekeeping/test_parser.py tests/timekeeping/test_tools.py`
Expected: FAIL because target text is not parsed/resolved and due-event delivery ignores routing policy.

- [ ] **Step 3: Parse target clauses without embedding device authority in timekeeping parser**
Extract phrases such as `on the bedroom speaker`, `in the kitchen`, and `on downstairs`; resolve them through `OutputRoutingService` in the canonical timekeeping tool before persistence.

- [ ] **Step 4: Route due timer/alarm events through `OutputDeliveryService`**
Preserve restart reconciliation and nearest-deadline scheduling. Delivery must not reclassify the timer/alarm action itself; it only resolves the authorized output route and reports delivery result/fallback truthfully.

- [ ] **Step 5: Run timekeeping regressions and commit**
Run: `pytest -q tests/timekeeping tests/output_routing/test_timekeeping_delivery.py tests/mobile_api/test_action_scope_enforcement.py`
Expected: PASS.
`git add rex/timekeeping rex/output_routing/delivery.py tests/timekeeping tests/output_routing/test_timekeeping_delivery.py && git commit -m "feat(routing): route timers and alarms to canonical outputs"`

### Task 4: Spoken-response/media execution integration

**Files:**
- Modify: `rex/media/service.py`
- Modify: `rex/voice/tts.py`
- Modify: `bridge/rex_voice_bridge.py`
- Modify: `rex/actions/dispatcher.py`
- Test: `tests/output_routing/test_voice_media_routing.py`

**Interfaces:**
- Media service asks `OutputRoutingService` for target/account when the command lacks an explicit selection.
- TTS delivery asks `OutputRoutingService.resolve(..., output_kind=OutputKind.SPOKEN_RESPONSE, ...)` before choosing local/smart-speaker output; text generation remains unchanged.

- [ ] **Step 1: Write failing request-origin and per-user spoken-output tests**

```python
def test_spoken_response_prefers_origin_when_authorized():
    route = routing.resolve(user_id="james", output_kind=OutputKind.SPOKEN_RESPONSE, explicit_target_text=None, origin_device_id="speaker_den", at=NOON)
    assert route.target_id == "local:speaker_den"
```

- [ ] **Step 2: Run focused tests and verify red state**
Run: `pytest -q tests/output_routing/test_voice_media_routing.py`
Expected: FAIL because voice/media delivery still chooses legacy/global output settings.

- [ ] **Step 3: Integrate routing decisions without moving conversation logic into output code**
Pass trusted `user_id`, origin device, and explicit target intent into routing; keep TTS synthesis and media provider action code independent. Apply temporary target volume immediately before delivery and restore previous device volume only when provider state/verification makes restoration safe.

- [ ] **Step 4: Run voice/media regressions and commit**
Run: `pytest -q tests/output_routing/test_voice_media_routing.py tests/test_us311_audio_output.py tests/test_us136_audio_playback.py tests/media`
Expected: PASS.
`git add rex/media/service.py rex/voice/tts.py bridge/rex_voice_bridge.py rex/actions/dispatcher.py tests/output_routing/test_voice_media_routing.py && git commit -m "feat(routing): apply per-user voice and media routes"`

### Task 5: Electron Settings and authenticated mobile settings use one backend

**Files:**
- Create: `bridge/rex_output_routing_bridge.py`
- Create: `gui/src/main/handlers/outputRouting.ts`
- Create: `gui/src/pages/settings/OutputRoutingSettingsSection.tsx`
- Modify: `gui/src/pages/SettingsPage.tsx`
- Modify: `gui/src/preload/index.ts`
- Modify: `gui/src/types/ipc.ts`
- Modify: `rex/mobile_api/grants.py`
- Modify: `rex/mobile_api/authorization.py`
- Create: `rex/mobile_api/routes/settings.py`
- Modify: `rex/mobile_api/routes/scaffolds.py`
- Modify: `rex/mobile_api/app.py`
- Test: `tests/output_routing/test_bridge.py`
- Test: `tests/mobile_api/test_output_routing_settings.py`
- Test: `tests/outputRoutingSettings.test.tsx`

**Interfaces:**
- Bridge/mobile operations: `get_policy`, `update_policy`, `list_targets`, `list_groups`, `list_media_accounts`, `set_default_media_account`, group CRUD/test playback.
- Add paired-device scopes `settings.read` and `settings.write`; writes remain user-bound and cannot edit another user's policy.

- [ ] **Step 1: Write failing bridge/mobile/renderer tests**

```python
def test_mobile_cannot_write_another_users_routing_policy(client, cole_token):
    response = client.put("/mobile/settings/output-routing", headers=cole_token, json={"user_id": "james", "media_target_id": "ha:den"})
    assert response.status_code == 403
```

- [ ] **Step 2: Run tests and verify red state**
Run: `pytest -q tests/output_routing/test_bridge.py tests/mobile_api/test_output_routing_settings.py`
Run: `cd gui && npm.cmd test -- --run tests/outputRoutingSettings.test.tsx`
Expected: FAIL because the shared routing settings surface does not exist.

- [ ] **Step 3: Implement transport-only Electron/mobile adapters**
Both surfaces instantiate/use the same `OutputRoutingService`; renderer/mobile payloads contain IDs and editable policy fields only. Media credentials remain in the vault and account lists expose safe display metadata.

- [ ] **Step 4: Build the Settings section around canonical IDs**
Show spoken/timer/alarm/media defaults, request-origin preference, conditions, quiet hours, fallback mode/target, target volume, media account/default provider, and group CRUD/test playback. Disabled/offline targets remain visible with truthful status rather than disappearing.

- [ ] **Step 5: Run desktop/mobile tests and commit**
Run: `pytest -q tests/output_routing/test_bridge.py tests/mobile_api/test_output_routing_settings.py tests/mobile_api/test_grant_enforcement.py`
Run: `cd gui && npm.cmd test -- --run tests/outputRoutingSettings.test.tsx && npm.cmd run typecheck && npm.cmd run build`
Expected: PASS.
`git add bridge/rex_output_routing_bridge.py gui/src rex/mobile_api tests/output_routing/test_bridge.py tests/mobile_api/test_output_routing_settings.py && git commit -m "feat(routing): expose shared output settings"`

### Task 6: US-122 documentation and release gates

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `PRD-production-readiness.md`
- Modify: `docs/superpowers/specs/2026-08-15-timers-alarms-media-routing.md`
- Modify: `docs/superpowers/specs/2026-08-16-situational-context-media-privacy-design.md` only if implementation evidence changes a documented boundary.

- [ ] **Step 1: Run the complete US-122 focused matrix**
Run: `pytest -q tests/output_routing tests/timekeeping tests/media tests/test_voice_identity_fallback.py tests/mobile_api/test_output_routing_settings.py tests/mobile_api/test_action_scope_enforcement.py`
Run: `cd gui && npm.cmd test -- --run tests/outputRoutingSettings.test.tsx && npm.cmd run typecheck && npm.cmd run build`
Expected: PASS.

- [ ] **Step 2: Run static/security gates**
Run: `ruff check rex/output_routing rex/media rex/timekeeping rex/runtime rex/mobile_api bridge/rex_output_routing_bridge.py tests/output_routing`
Run: `black --check rex/output_routing rex/media rex/timekeeping rex/runtime rex/mobile_api bridge/rex_output_routing_bridge.py tests/output_routing`
Run: `mypy rex/output_routing rex/media rex/timekeeping rex/runtime rex/mobile_api --ignore-missing-imports`
Run: `python scripts/security_audit.py --release-gate`
Run: `pre-commit run --all-files`
Run: `git diff --check`
Expected: all PASS.

- [ ] **Step 3: Update docs/tracker with proven behavior only**
Mark each US-122 criterion only when executable evidence exists. Do not claim Apple Music connected/authenticated, native iOS delivery, or physical-speaker behavior that has not been exercised.

- [ ] **Step 4: Commit acceptance evidence**
`git add README.md CLAUDE.md PRD-production-readiness.md docs/superpowers/specs && git commit -m "docs(routing): record US-122 acceptance evidence"`

## US-122 Completion Check

Before the implementation PR is mergeable, confirm James/Cole-equivalent test users can hold different provider accounts and routing policies concurrently, unresolved-speaker primary-account fallback cannot mutate a private library, explicit targets override stored policy, and unavailable targets follow the configured fallback rather than silently rerouting.
