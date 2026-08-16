# Task 2 Implementation Report

## Status

DONE

## Implementation commit

`c9440adbe36d6f164688527c516c7d820d5dd861` (`feat(media): adapt providers and persist speaker groups`)

## Files changed

- `rex/media/adapters.py`: Added canonical Smart Speaker, Home Assistant, and Music Assistant provider adapters.
- `rex/media/groups.py`: Added atomic household speaker-group persistence and mixed-provider capability intersection.
- `rex/media/__init__.py`: Exported the Task 2 adapters and group contracts.
- `rex/audio/speaker_discovery.py`: Added the provider-owned canonical target ID for discovered Sonos/Bose speakers.
- `rex/integrations/music_assistant.py`: Declared the adapter operations backed by the client's existing methods. No discovery or state API was added.
- `rex/ha_bridge.py`: Preserved full HA entity state in `list_entities()` and exposed the existing Rex device-state reader through `get_entity_state()`.
- `tests/media/test_provider_adapters.py`: Added focused provider discovery, mutation, state, and truthful-unsupported tests.
- `tests/media/test_groups.py`: Added group validation, persistence, capability intersection, mutation, and atomic-failure tests.

## RED evidence

Exact command:

```powershell
& 'C:\Users\james\rex-ai-test\rex-ai-assistant\.venv\Scripts\python.exe' -m pytest -q tests/media/test_provider_adapters.py tests/media/test_groups.py
```

Expected RED result: exit code 1 during collection with two errors and zero collected tests:

- `ModuleNotFoundError: No module named 'rex.media.adapters'`
- `ModuleNotFoundError: No module named 'rex.media.groups'`

A later provider-ownership RED cycle used the same command and produced two focused failures before production changes:

- `AttributeError: 'DiscoveredSpeaker' object has no attribute 'target_id'`
- Music Assistant play was incorrectly accepted when the client-declared support set excluded play.

## Design choices

- Provider discovery remains provider-neutral. None of the adapters accepts a user ID or makes an authorization decision. Authorization stays in `AudioTargetRegistry`.
- `SmartSpeakerAdapter` wraps only `SpeakerDiscoveryService.get_cached_speakers()`. Sonos/Bose targets advertise no canonical media mutations, return non-acceptance for actions, and return `UNKNOWN` state.
- `MusicAssistantAdapter` wraps only the existing `play`, `pause`, `resume`, `skip`, and `set_volume` client methods. It discovers no targets and returns `UNKNOWN` state because the existing client cannot independently read either. No new Music Assistant HTTP route or external API assumption was added.
- `HomeAssistantMediaAdapter` filters `HABridge.list_entities()` to `media_player.*`, maps only the media actions already exposed in `rex/routes/ha.py`, uses the existing HA intent execution path, and reads state independently through the existing `rex.ha.device_state` path.
- Provider acknowledgement is never treated as verified postcondition state. Unsupported actions return explicit non-acceptance.
- `SpeakerGroupStore` defaults to `household_data_path("media", "speaker_groups.json")`, persists generated `group:<uuid>` IDs and member target IDs, validates every proposed member, rejects empty and duplicate members, normalizes names for case-insensitive duplicate detection, and blocks rename/set-members mutations that would retain unresolved members.
- Group capabilities are calculated from the current member capability intersection and are not persisted as potentially stale provider state.
- JSON writes use a same-directory temporary file, flush plus `fsync`, and `os.replace`; temporary files are removed after failures.

## GREEN and regression evidence

Focused GREEN:

```powershell
& 'C:\Users\james\rex-ai-test\rex-ai-assistant\.venv\Scripts\python.exe' -m pytest -q tests/media/test_provider_adapters.py tests/media/test_groups.py
```

Result: `18 passed in 0.46s`.

Required provider/group regression:

```powershell
& 'C:\Users\james\rex-ai-test\rex-ai-assistant\.venv\Scripts\python.exe' -m pytest -q tests/media/test_provider_adapters.py tests/media/test_groups.py tests/test_speaker_discovery.py tests/test_sp002_smart_speaker_output.py tests/test_us021_music_assistant.py
```

Result: `52 passed in 1.40s`.

Additional Task 1 and HA safety regression:

```powershell
& 'C:\Users\james\rex-ai-test\rex-ai-assistant\.venv\Scripts\python.exe' -m pytest -q tests/media/test_target_registry.py tests/test_us042_ha_api_connection.py tests/test_us028_device_state.py
```

Result: `44 passed in 0.82s`.

Ruff:

```powershell
& 'C:\Users\james\rex-ai-test\rex-ai-assistant\.venv\Scripts\python.exe' -m ruff check rex/audio/speaker_discovery.py rex/ha_bridge.py rex/integrations/music_assistant.py rex/media/__init__.py rex/media/adapters.py rex/media/groups.py tests/media/test_provider_adapters.py tests/media/test_groups.py
```

Result: `All checks passed!`

mypy:

```powershell
& 'C:\Users\james\rex-ai-test\rex-ai-assistant\.venv\Scripts\python.exe' -m mypy rex/audio/speaker_discovery.py rex/ha_bridge.py rex/integrations/music_assistant.py rex/media/__init__.py rex/media/adapters.py rex/media/groups.py tests/media/test_provider_adapters.py tests/media/test_groups.py
```

Result: `Success: no issues found in 8 source files` (with the repository's existing unused-mypy-section note).

Black:

```powershell
& 'C:\Users\james\rex-ai-test\rex-ai-assistant\.venv\Scripts\python.exe' -m black --check rex/audio/speaker_discovery.py rex/ha_bridge.py rex/integrations/music_assistant.py rex/media/__init__.py rex/media/adapters.py rex/media/groups.py tests/media/test_provider_adapters.py tests/media/test_groups.py
```

Result: `8 files would be left unchanged`.

Diff validation:

```powershell
git diff --check
```

Result: exit code 0 with no whitespace errors. Git emitted only the repository's line-ending conversion warnings.

## Self-review findings and fixes

- Found that adapter support and smart-speaker ID formatting initially lived only in the adapter. Added a fresh failing test cycle, moved the canonical ID onto `DiscoveredSpeaker`, and made `MusicAssistantClient` declare the existing operations available to its adapter.
- Found overly concrete adapter constructor annotations that rejected interface-faithful fakes under mypy. Replaced them with narrow structural protocols matching the existing provider methods.
- Found `list` type annotations colliding with the required `SpeakerGroupStore.list()` method under mypy. Added module-level record aliases and reran all gates.
- Found one duplicate test import and Black formatting differences. Removed/formatted them, then reran focused tests, regressions, Ruff, mypy, Black, and diff validation.
- Confirmed no Task 3+ files or behavior were read or implemented, and no unrelated files were modified.

## Concerns or blockers

None.
