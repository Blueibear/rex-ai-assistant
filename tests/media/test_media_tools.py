from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rex.capabilities.registry import CapabilityRegistry
from rex.media.accounts import MediaAccountStore
from rex.media.models import (
    AudioTarget,
    MediaAction,
    MediaActionAcknowledgement,
    MediaCapability,
    MediaState,
    MediaStateSnapshot,
    TargetKind,
)
from rex.media.registry import AudioTargetRegistry
from rex.media.service import MediaService
from rex.media.sessions import ActiveMediaSessionStore
from rex.media.tools import media_manage, set_media_service, verify_media_mutation
from rex.tools.dispatcher import ToolDispatcher
from rex.tools.registry import _build_default_registry

_ALL_CAPABILITIES = frozenset(
    {
        MediaCapability.PLAY,
        MediaCapability.PAUSE,
        MediaCapability.RESUME,
        MediaCapability.STOP,
        MediaCapability.NEXT,
        MediaCapability.SET_VOLUME,
    }
)


def _target(target_id: str, *, display_name: str, room: str | None = None) -> AudioTarget:
    provider, native_id = target_id.split(":", 1)
    return AudioTarget(
        id=target_id,
        native_id=native_id,
        provider=provider,
        kind=TargetKind.SPEAKER,
        display_name=display_name,
        aliases=(),
        room=room,
        capabilities=_ALL_CAPABILITIES,
        online=True,
        health="healthy",
    )


class FakeAdapter:
    """In-memory provider double whose reads truthfully reflect applied state."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self._state: dict[str, MediaStateSnapshot] = {}
        self.calls: list[tuple[str, MediaAction, object]] = []
        self.ignore_actions: set[MediaAction] = set()

    def seed(self, target_id: str, snapshot: MediaStateSnapshot) -> None:
        self._state[target_id] = snapshot

    def discover_targets(self) -> tuple[AudioTarget, ...]:
        return ()

    def execute_action(
        self, target: AudioTarget, action: MediaAction, *, value: object = None
    ) -> MediaActionAcknowledgement:
        self.calls.append((target.id, action, value))
        if action not in self.ignore_actions:
            self._apply(target.id, action, value)
        return MediaActionAcknowledgement(accepted=True, detail="provider accepted")

    def get_state(self, target: AudioTarget) -> MediaStateSnapshot:
        return self._state.get(
            target.id,
            MediaStateSnapshot(
                target_id=target.id, playback=MediaState.UNKNOWN, observed_at=datetime.now(tz=UTC)
            ),
        )

    def _apply(self, target_id: str, action: MediaAction, value: object) -> None:
        current = self._state.get(
            target_id,
            MediaStateSnapshot(
                target_id=target_id, playback=MediaState.IDLE, observed_at=datetime.now(tz=UTC)
            ),
        )
        if action is MediaAction.PAUSE:
            playback = MediaState.PAUSED
        elif action in (MediaAction.PLAY, MediaAction.RESUME):
            playback = MediaState.PLAYING
        elif action is MediaAction.STOP:
            playback = MediaState.STOPPED
        else:
            playback = current.playback
        volume = current.volume_percent
        if action is MediaAction.SET_VOLUME and value is not None:
            volume = (
                float(value)
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else volume
            )
        self._state[target_id] = MediaStateSnapshot(
            target_id=target_id,
            playback=playback,
            observed_at=datetime.now(tz=UTC),
            volume_percent=volume,
        )


@pytest.fixture
def kitchen() -> AudioTarget:
    return _target("ha:media_player.kitchen", display_name="Kitchen Speaker", room="Kitchen")


@pytest.fixture
def wired_service(kitchen, tmp_path):
    adapter = FakeAdapter("ha")
    registry = AudioTargetRegistry([kitchen], authorized_target_ids={"james": {kitchen.id}})
    service = MediaService(
        registry=registry,
        adapters={"ha": adapter},
        account_store=MediaAccountStore(root=tmp_path),
        session_store=ActiveMediaSessionStore(ttl_seconds=300, clock=lambda: 1000.0),
        clock=lambda: 1000.0,
    )
    set_media_service(service)
    yield service, adapter
    set_media_service(None)


def test_media_tools_have_canonical_security_metadata() -> None:
    registry = _build_default_registry(capability_registry=CapabilityRegistry())
    read = registry.get("media_read")
    manage = registry.get("media_manage")

    assert read is not None
    assert read.operation == "read"
    assert read.requires_identity is True
    assert "ha_control" in read.required_permissions

    assert manage is not None
    assert manage.operation == "mutation"
    assert manage.requires_identity is True
    assert manage.verifier is not None
    assert "ha_control" in manage.required_permissions

    read_card = registry.capability_registry.get("media_read")
    manage_card = registry.capability_registry.get("media_manage")
    assert read_card is not None and read_card.verification_supported is False
    assert manage_card is not None and manage_card.verification_supported is True


def test_manage_tool_pause_is_reported_verified_through_the_dispatcher(
    wired_service, kitchen
) -> None:
    service, adapter = wired_service
    adapter.seed(
        kitchen.id,
        MediaStateSnapshot(
            target_id=kitchen.id, playback=MediaState.PLAYING, observed_at=datetime.now(tz=UTC)
        ),
    )
    registry = _build_default_registry(capability_registry=CapabilityRegistry())
    dispatcher = ToolDispatcher(registry)

    result = dispatcher.dispatch(
        "media_manage",
        {"action": "pause", "target_text": kitchen.id},
        {
            "user_id": "james",
            "request_id": "pause-1",
            "granted_permissions": frozenset({"ha_control"}),
        },
    )

    assert result.status == "verified"
    assert result.success is True
    assert adapter.calls == [(kitchen.id, MediaAction.PAUSE, None)]


def test_manage_tool_never_self_promotes_an_unverified_mutation(wired_service, kitchen) -> None:
    service, adapter = wired_service
    adapter.ignore_actions = {MediaAction.PAUSE}
    adapter.seed(
        kitchen.id,
        MediaStateSnapshot(
            target_id=kitchen.id, playback=MediaState.PLAYING, observed_at=datetime.now(tz=UTC)
        ),
    )
    registry = _build_default_registry(capability_registry=CapabilityRegistry())
    dispatcher = ToolDispatcher(registry)

    result = dispatcher.dispatch(
        "media_manage",
        {"action": "pause", "target_text": kitchen.id},
        {
            "user_id": "james",
            "request_id": "pause-2",
            "granted_permissions": frozenset({"ha_control"}),
        },
    )

    assert result.status == "attempted_unverified"
    assert result.success is False


def test_manage_tool_reports_failed_for_unsupported_action(wired_service, kitchen) -> None:
    service, adapter = wired_service
    registry = _build_default_registry(capability_registry=CapabilityRegistry())
    dispatcher = ToolDispatcher(registry)

    result = dispatcher.dispatch(
        "media_manage",
        {"action": "mute", "target_text": kitchen.id},
        {
            "user_id": "james",
            "request_id": "mute-1",
            "granted_permissions": frozenset({"ha_control"}),
        },
    )

    assert result.status == "failed"
    assert result.success is False
    assert adapter.calls == []


def test_manage_tool_requires_ha_control_permission(wired_service, kitchen) -> None:
    registry = _build_default_registry(capability_registry=CapabilityRegistry())
    dispatcher = ToolDispatcher(registry)

    result = dispatcher.dispatch(
        "media_manage",
        {"action": "pause", "target_text": kitchen.id},
        {"user_id": "james", "request_id": "pause-3", "granted_permissions": frozenset()},
    )

    assert result.status == "denied"


def test_manage_tool_rejects_read_actions(wired_service, kitchen) -> None:
    registry = _build_default_registry(capability_registry=CapabilityRegistry())
    dispatcher = ToolDispatcher(registry)

    result = dispatcher.dispatch(
        "media_manage",
        {"action": "state", "target_text": kitchen.id},
        {
            "user_id": "james",
            "request_id": "bad-1",
            "granted_permissions": frozenset({"ha_control"}),
        },
    )

    assert result.status == "failed"


def test_read_tool_reports_current_state(wired_service, kitchen) -> None:
    service, adapter = wired_service
    adapter.seed(
        kitchen.id,
        MediaStateSnapshot(
            target_id=kitchen.id,
            playback=MediaState.PLAYING,
            observed_at=datetime.now(tz=UTC),
            volume_percent=55.0,
        ),
    )
    registry = _build_default_registry(capability_registry=CapabilityRegistry())
    dispatcher = ToolDispatcher(registry)

    result = dispatcher.dispatch(
        "media_read",
        {"action": "state", "target_text": kitchen.id},
        {
            "user_id": "james",
            "request_id": "read-1",
            "granted_permissions": frozenset({"ha_control"}),
        },
    )

    assert result.success is True
    assert result.output["state"]["playback"] == "playing"
    assert result.output["state"]["volume_percent"] == 55.0


def test_read_tool_rejects_mutation_actions(wired_service, kitchen) -> None:
    registry = _build_default_registry(capability_registry=CapabilityRegistry())
    dispatcher = ToolDispatcher(registry)

    result = dispatcher.dispatch(
        "media_read",
        {"action": "pause", "target_text": kitchen.id},
        {
            "user_id": "james",
            "request_id": "read-2",
            "granted_permissions": frozenset({"ha_control"}),
        },
    )

    assert result.success is False


def test_direct_calls_use_transcript_parsing(wired_service, kitchen) -> None:
    service, adapter = wired_service
    adapter.seed(
        kitchen.id,
        MediaStateSnapshot(
            target_id=kitchen.id, playback=MediaState.PLAYING, observed_at=datetime.now(tz=UTC)
        ),
    )

    output = media_manage(
        transcript=f"pause the {kitchen.native_id}",
        target_text=kitchen.id,
        _user_id="james",
    )
    assert output["lifecycle_state"] == "verified"


def test_verify_media_mutation_reads_live_state_independently(wired_service, kitchen) -> None:
    service, adapter = wired_service
    adapter.seed(
        kitchen.id,
        MediaStateSnapshot(
            target_id=kitchen.id, playback=MediaState.PAUSED, observed_at=datetime.now(tz=UTC)
        ),
    )
    args: dict[str, object] = {}
    output = {
        "verification": {
            "target_id": kitchen.id,
            "provider": "ha",
            "user_id": "james",
            "expected": {"playback": ["paused"]},
        }
    }

    assert verify_media_mutation(args, output) is True

    output["verification"]["expected"] = {"playback": ["playing"]}
    assert verify_media_mutation(args, output) is False


@pytest.mark.parametrize(
    "output",
    [
        None,
        {},
        {"verification": "not-a-dict"},
        {"verification": {}},
        {"verification": {"target_id": "x", "provider": "ha", "user_id": "james", "expected": {}}},
    ],
)
def test_verify_media_mutation_fails_closed_on_malformed_output(wired_service, output) -> None:
    assert verify_media_mutation({}, output) is False
