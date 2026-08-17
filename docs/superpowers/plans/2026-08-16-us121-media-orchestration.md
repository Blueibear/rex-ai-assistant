# US-121 Canonical Media Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build provider-neutral speaker/room/group discovery and verified media control, with user-bound provider accounts, request-origin default routing, and bounded active-media follow-up context.

**Architecture:** Add a focused `rex.media` package above existing speaker discovery, Home Assistant, Music Assistant, credential-vault, and ToolExecutionLifecycle code. Canonical target/account/session models remain provider-neutral; adapters translate provider-specific discovery/actions, and one media service resolves authorization, target, account, action, and verification. The existing direct `MusicHandler` remains only as a compatibility parser during migration and is removed from direct dispatch after canonical media tools pass regression tests.

**Tech Stack:** Python 3.11, Pydantic/dataclasses, existing `rex.capabilities`/`rex.tools` lifecycle, Windows DPAPI credential vault, Home Assistant, Music Assistant, Electron bridge/TypeScript, pytest.

## Global Constraints

- No new heavy dependency and no provider-specific user command grammar.
- Apple Music/MusicKit is an adapter contract only until Apple developer credentials and real user authorization exist; do not claim live Apple Music support.
- A display-name, room-name, or request-origin match never grants device/account authority.
- Media mutations report success only after canonical independent verification when the provider exposes verifiable state; otherwise return attempted/unverified truthfully.
- Provider-account ownership and output-target authorization are separate decisions.
- Ambiguous target/session resolution asks one short clarification rather than fuzzy-guessing.
- Preserve current Sonos/Bose discovery and Music Assistant behavior through adapters while removing the legacy direct-execution bypass.

---

### Task 1: Canonical media models and target registry

**Files:**
- Create: `rex/media/__init__.py`
- Create: `rex/media/models.py`
- Create: `rex/media/registry.py`
- Test: `tests/media/test_target_registry.py`

**Interfaces:**
- Produces: `AudioTarget`, `TargetKind`, `MediaCapability`, `MediaAction`, `MediaState`, `TargetResolution`, `TargetProviderAdapter`, `AudioTargetRegistry`.
- `AudioTargetRegistry.resolve(query: str | None, *, user_id: str, origin_device_id: str | None = None) -> TargetResolution` is the only target-name/origin resolution API used by later tasks.

- [ ] **Step 1: Write failing registry tests for exact aliases, room uniqueness, ambiguity, authorization filtering, and origin-device default**

```python
def test_origin_is_only_a_preference_after_authorization():
    registry = make_registry(origin_target_authorized=False)
    result = registry.resolve(None, user_id="james", origin_device_id="mic_kitchen")
    assert result.target is None
    assert result.reason == "origin_not_authorized"


def test_ambiguous_room_does_not_guess():
    registry = make_registry(two_living_room_targets=True)
    result = registry.resolve("living room", user_id="james")
    assert result.target is None
    assert result.ambiguous_ids == ("ha:media_player.living_room", "sonos:RINCON_2")
```

- [ ] **Step 2: Run the focused test and verify red state**

Run: `pytest -q tests/media/test_target_registry.py`
Expected: FAIL because `rex.media` and `AudioTargetRegistry` do not exist.

- [ ] **Step 3: Implement immutable canonical models and deterministic resolver**

```python
@dataclass(frozen=True, slots=True)
class AudioTarget:
    id: str
    native_id: str
    provider: str
    kind: TargetKind
    display_name: str
    aliases: tuple[str, ...]
    room: str | None
    capabilities: frozenset[MediaCapability]
    online: bool
    health: str
```

Resolution order: explicit stable ID -> exact normalized name/alias -> exact room if unique -> persistent group -> trusted origin-device mapping when query is absent. Filter unauthorized/offline targets before returning a match; never substring/fuzzy-select an ambiguous result.

- [ ] **Step 4: Run focused tests to green**
Run: `pytest -q tests/media/test_target_registry.py`
Expected: PASS.

- [ ] **Step 5: Commit**
`git add rex/media tests/media/test_target_registry.py && git commit -m "feat(media): add canonical audio target registry"`

### Task 2: Provider adapters and persistent speaker groups

**Files:**
- Create: `rex/media/adapters.py`
- Create: `rex/media/groups.py`
- Modify: `rex/audio/speaker_discovery.py`
- Modify: `rex/integrations/music_assistant.py`
- Modify: `rex/ha_bridge.py`
- Test: `tests/media/test_provider_adapters.py`
- Test: `tests/media/test_groups.py`

**Interfaces:**
- Consumes: `AudioTarget`, `MediaAction`, `MediaState`, `TargetProviderAdapter`.
- Produces: `SmartSpeakerAdapter`, `MusicAssistantAdapter`, `HomeAssistantMediaAdapter`, `SpeakerGroupStore` with `create/get/list/rename/set_members/delete`.

- [ ] **Step 1: Write failing adapter/group tests**

```python
def test_ha_adapter_discovers_only_media_players():
    adapter = HomeAssistantMediaAdapter(fake_ha(states=[light_state(), media_state("media_player.den")]))
    targets = adapter.discover(user_id="james")
    assert [target.native_id for target in targets] == ["media_player.den"]


def test_group_rejects_unknown_member(tmp_path):
    store = SpeakerGroupStore(tmp_path / "groups.json", target_exists=lambda target_id: False)
    with pytest.raises(ValueError, match="Unknown audio target"):
        store.create("Downstairs", ["missing:target"])
```

- [ ] **Step 2: Run focused tests and verify they fail**
Run: `pytest -q tests/media/test_provider_adapters.py tests/media/test_groups.py`
Expected: FAIL on missing adapter/group classes.

- [ ] **Step 3: Implement adapters by wrapping existing provider code, not duplicating it**
`SmartSpeakerAdapter` consumes `SpeakerDiscoveryService`; `HomeAssistantMediaAdapter` consumes `HABridge.list_entities()` plus the existing HA mutation/state APIs; `MusicAssistantAdapter` wraps `MusicAssistantClient`. Provider capability sets are derived from real supported operations, so Sonos/Bose discovery may expose output/test capabilities without falsely advertising search/transfer controls they cannot perform.

- [ ] **Step 4: Implement atomic household group persistence**
Persist stable group IDs and member target IDs under the canonical household data root. Reject duplicate names after casefold normalization, unknown members, empty groups, and mutations that leave unresolved members; preserve mixed-provider groups but expose only the capability intersection.

- [ ] **Step 5: Run provider/group regressions**
Run: `pytest -q tests/media/test_provider_adapters.py tests/media/test_groups.py tests/test_speaker_discovery.py tests/test_sp002_smart_speaker_output.py tests/test_us021_music_assistant.py`
Expected: PASS.

- [ ] **Step 6: Commit**
`git add rex/media rex/audio/speaker_discovery.py rex/integrations/music_assistant.py rex/ha_bridge.py tests/media tests/test_speaker_discovery.py tests/test_us021_music_assistant.py && git commit -m "feat(media): adapt providers and persist speaker groups"`

### Task 3: User-bound media accounts and active media sessions

**Files:**
- Create: `rex/media/accounts.py`
- Create: `rex/media/sessions.py`
- Test: `tests/media/test_accounts.py`
- Test: `tests/media/test_sessions.py`

**Interfaces:**
- Produces: `MediaAccountRef`, `MediaAccountStore`, `ActiveMediaSession`, `ActiveMediaSessionStore`.
- `MediaAccountStore.put(user_id: str, provider: str, account_id: str, credential_ref: str, display_name: str) -> MediaAccountRef` stores metadata only; secrets remain in `get_credential_vault(scope="user", user_id=user_id)`.
- `ActiveMediaSessionStore.set(session: ActiveMediaSession)`, `get(user_id: str, *, now: float | None = None)`, and `clear(user_id: str)` are explicitly user-keyed and TTL-bound.

- [ ] **Step 1: Write failing account-isolation and session-expiry tests**

```python
def test_account_lookup_cannot_cross_user(tmp_path):
    store = MediaAccountStore(tmp_path)
    store.put("james", "apple_music", "main", "cred_j", "James Apple Music")
    assert store.get("cole", "apple_music", "main") is None


def test_active_session_expires():
    store = ActiveMediaSessionStore(ttl_seconds=300, clock=lambda: 1000.0)
    store.set(ActiveMediaSession(user_id="james", target_id="ha:den", provider="ha", media_ref="track:1", updated_at=600.0))
    assert store.get("james") is None
```

- [ ] **Step 2: Run focused tests and confirm failure**
Run: `pytest -q tests/media/test_accounts.py tests/media/test_sessions.py`
Expected: FAIL on missing stores.

- [ ] **Step 3: Implement metadata-only account storage and bounded session state**
Reject invalid user IDs, empty provider/account IDs, credential refs that are not syntactically valid, and session reads for another user. Do not put tokens, playlist contents, or transcript text in session records.

- [ ] **Step 4: Add Apple Music as a declared provider capability contract only**
Represent `apple_music` in provider/account metadata and capability tests; do not add a fake token flow, MusicKit call, or “connected” status without real developer/user credentials.

- [ ] **Step 5: Run focused tests to green**
Run: `pytest -q tests/media/test_accounts.py tests/media/test_sessions.py tests/test_credential_vault.py`
Expected: PASS.

- [ ] **Step 6: Commit**
`git add rex/media/accounts.py rex/media/sessions.py tests/media/test_accounts.py tests/media/test_sessions.py && git commit -m "feat(media): isolate provider accounts and active sessions"`

### Task 4: Canonical media service, parser, tools, and verification

**Files:**
- Create: `rex/media/parser.py`
- Create: `rex/media/service.py`
- Create: `rex/media/tools.py`
- Modify: `rex/tools/registry.py`
- Modify: `rex/local_tool_executor.py`
- Test: `tests/media/test_media_parser.py`
- Test: `tests/media/test_media_service.py`
- Test: `tests/media/test_media_tools.py`

**Interfaces:**
- Produces: `MediaCommand(action, query, target_text, level)`, `parse_media_command(text)`, `MediaService.execute(command, *, user_id, origin_device_id, account_ref=None)`, `media_read`, `media_manage`, `verify_media_mutation`.
- `MediaService` is the only component allowed to combine account lookup, target resolution, provider dispatch, session update, and state verification.

- [ ] **Step 1: Write failing natural-language and verified-mutation tests**

```python
def test_move_it_uses_active_session_target_context():
    service = service_with_active_session("james", target_id="ha:kitchen")
    result = service.execute(parse_media_command("move it to the living room"), user_id="james", origin_device_id="mic_kitchen")
    assert result.requested_target_id == "ha:living_room"


def test_unverified_provider_mutation_is_not_success():
    result = media_manage(action="pause", user_id="james", target_id="provider:x")
    assert result["lifecycle_state"] == "unverified"
```

- [ ] **Step 2: Run focused tests and confirm red state**
Run: `pytest -q tests/media/test_media_parser.py tests/media/test_media_service.py tests/media/test_media_tools.py`
Expected: FAIL on missing parser/service/tools.

- [ ] **Step 3: Implement deterministic common media grammar**
Recognize play/pause/resume/stop/next/previous, volume, mute/unmute, state query, and transfer phrases. Parse a target phrase only; never resolve provider/device authority in the parser.

- [ ] **Step 4: Implement service execution and verification**
Dispatch only after target/account authorization. For mutations, independently re-read provider state where supported and compare the expected state; update active session only after the action outcome is at least truthfully known. A provider transport success without matching state remains unverified.

- [ ] **Step 5: Replace delegated registry entries with executable canonical `media_read` and `media_manage` cards**
Use identity-required user context, explicit media/home-control permission policy, mutation metadata, and `verify_media_mutation`; retire `music_*` delegated handlers only after compatibility routing in Task 5 is green.

- [ ] **Step 6: Run canonical tool/lifecycle regressions**
Run: `pytest -q tests/media/test_media_parser.py tests/media/test_media_service.py tests/media/test_media_tools.py tests/test_tools_registry.py tests/test_tool_execution_lifecycle.py tests/rex2/test_action_lifecycle.py`
Expected: PASS.

- [ ] **Step 7: Commit**
`git add rex/media rex/tools/registry.py rex/local_tool_executor.py tests/media tests/test_tools_registry.py && git commit -m "feat(media): route controls through canonical lifecycle"`

### Task 5: TurnEngine integration, request-origin routing, and legacy music migration

**Files:**
- Modify: `rex/actions/dispatcher.py`
- Modify: `rex/assistant.py`
- Modify: `rex/music_handler.py`
- Modify: `rex/runtime/invocation.py` only if a trusted origin-device helper is needed; do not duplicate `TurnContext.device_id`.
- Test: `tests/media/test_media_turn_routing.py`
- Modify/Test: `tests/test_us022_music_handler.py`
- Modify/Test: `tests/test_us024_speaker_origin.py`
- Modify/Test: `tests/test_us016_action_dispatcher.py`

**Interfaces:**
- Consumes: `parse_media_command`, canonical `media_read/media_manage`, existing `current_turn_invocation().device_id`.
- Produces: one exact pre-LLM media dispatch path analogous to timekeeping; no direct provider mutation from `MusicHandler`.

- [ ] **Step 1: Write failing end-to-end turn tests**

```python
@pytest.mark.asyncio
async def test_play_without_target_uses_trusted_origin(dispatcher):
    with turn_invocation(TurnSource.VOICE, device_id="mic_kitchen"):
        result = await dispatcher.dispatch(intent, context, "play jazz", user_id="james")
    assert result.response == "Playing jazz in the kitchen."


@pytest.mark.asyncio
async def test_followup_move_it_does_not_fan_out(dispatcher):
    result = await dispatcher.dispatch(intent, context, "move it to the living room", user_id="james")
    assert dispatcher.tool_dispatcher.calls == ["media_manage"]
```

- [ ] **Step 2: Run turn routing tests and verify failure**
Run: `pytest -q tests/media/test_media_turn_routing.py tests/test_us016_action_dispatcher.py`
Expected: FAIL because media still uses the direct `MusicHandler` path.

- [ ] **Step 3: Add exact media routing before generic capability retrieval**
Mirror the timekeeping pattern: parse once, select exactly `media_read` or `media_manage`, pass validated `user_id` plus trusted `origin_device_id`, and prevent HA/generic multi-tool fanout for the same media command.

- [ ] **Step 4: Reduce `MusicHandler` to compatibility parsing/deprecation or remove its ActionDispatcher injection**
No media mutation may bypass `ToolExecutionLifecycle`. Preserve old phrase tests by routing the same phrases through `parse_media_command` and canonical tools.

- [ ] **Step 5: Run integration regressions**
Run: `pytest -q tests/media tests/test_us022_music_handler.py tests/test_us024_speaker_origin.py tests/test_us016_action_dispatcher.py tests/test_assistant.py tests/test_tools_registry.py`
Expected: PASS.

- [ ] **Step 6: Commit**
`git add rex/actions/dispatcher.py rex/assistant.py rex/music_handler.py rex/runtime/invocation.py tests/media tests/test_us022_music_handler.py tests/test_us024_speaker_origin.py tests/test_us016_action_dispatcher.py && git commit -m "feat(media): integrate canonical conversational routing"`

### Task 6: Canonical discovery/group bridge and US-121 acceptance gates

**Files:**
- Modify: `bridge/rex_speaker_bridge.py`
- Modify: `rex/assistant.py`
- Modify: `rex/media/service.py`
- Modify: `rex/media/models.py`
- Modify: `rex/media/adapters.py`
- Modify: `rex/ha_bridge.py`
- Modify: `gui/src/main/handlers/speakers.ts`
- Modify: `gui/src/main/ipc.ts`
- Modify: `gui/src/preload/index.ts`
- Modify: `gui/src/pages/settings/AudioOutputSettingsSection.tsx`
- Modify: `gui/src/types/ipc.ts`
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `PRD-production-readiness.md`
- Modify: `docs/superpowers/specs/2026-08-15-timers-alarms-media-routing.md`
- Test: `tests/media/test_speaker_bridge.py`
- Test: `gui/tests/speakerHandlers.test.ts`

**Interfaces:**
- Bridge commands: `list_targets`, `refresh_targets`, `list_groups`, `create_group`, `rename_group`, `set_group_members`, `delete_group`.
- Responses expose IDs/names/provider/room/capabilities/health only; never credential refs or private account tokens.

- [x] **Step 1: Write failing bridge tests for canonical target/group payloads and refresh**

```python
def test_list_targets_does_not_expose_credentials(fake_registry):
    body, code = handle_speaker_request({"command": "list_targets", "user_id": "james"}, registry=fake_registry)
    assert code == 0
    assert "credential_ref" not in json.dumps(body)
```

- [x] **Step 2: Run bridge tests and verify red state**
Run: `pytest -q tests/media/test_speaker_bridge.py`
Expected: FAIL because the bridge only supports legacy `list` discovery.

- [x] **Step 3: Migrate bridge/IPC to canonical registry and group store**
Keep renderer handlers transport-only. Authenticate/bind the active user before private authorization filtering; do not accept request-supplied authority to view another user's restricted targets.

- [x] **Step 4: Run Python and GUI focused tests**
Run: `pytest -q tests/media tests/test_speaker_discovery.py tests/test_us021_music_assistant.py tests/test_us022_music_handler.py tests/test_tools_registry.py`
Run: `cd gui && npm.cmd test -- --run tests/speakerHandlers.test.ts && npm.cmd run typecheck && npm.cmd run build`
Expected: all PASS.

- [x] **Step 5: Update tracker/docs only for behavior proven by tests**
Document canonical targets/groups/media tools, request-origin preference, provider limitations, and Apple Music as planned adapter-only. Leave physical-provider acceptance explicitly unverified until a real speaker path is exercised.

- [x] **Step 6: Run release-quality gates**
Run: `ruff check rex/media rex/actions/dispatcher.py rex/tools/registry.py bridge/rex_speaker_bridge.py tests/media`
Run: `black --check rex/media rex/actions/dispatcher.py rex/tools/registry.py bridge/rex_speaker_bridge.py tests/media`
Run: `mypy rex/media rex/actions/dispatcher.py rex/tools/registry.py --ignore-missing-imports`
Run: `python scripts/security_audit.py --release-gate`
Run: `pre-commit run --all-files`
Run: `git diff --check`
Expected: all gates PASS and working tree contains only intended changes.

- [x] **Step 7: Commit**
`git add bridge/rex_speaker_bridge.py rex/assistant.py rex/media/service.py rex/media/models.py rex/media/adapters.py rex/media/tools.py rex/ha_bridge.py gui/src/main/handlers/speakers.ts gui/src/main/ipc.ts gui/src/preload/index.ts gui/src/pages/settings/AudioOutputSettingsSection.tsx gui/src/types/ipc.ts README.md CLAUDE.md PRD-production-readiness.md docs/superpowers/specs/2026-08-15-timers-alarms-media-routing.md docs/superpowers/plans/2026-08-16-us121-media-orchestration.md tests/media tests/rex2/test_capability_registry.py gui/tests/speakerHandlers.test.ts && git commit -m "feat(media): complete canonical speaker orchestration"`

## US-121 Completion Check

Before opening the implementation PR, run the complete focused matrix above plus the repository release pytest matrix. Record exact counts, leave the GitHub-check checkbox open until the exact PR head is green, and do not claim Apple Music or physical-speaker production verification without real credentials/hardware evidence.
