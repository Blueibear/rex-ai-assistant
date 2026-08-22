from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rex.context.active import ActiveContextStore
from rex.credential_vault import generate_credential_ref
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
from rex.media.parser import MediaCommand
from rex.media.registry import AudioTargetRegistry
from rex.media.service import MediaService
from rex.media.sessions import ActiveMediaSession, ActiveMediaSessionStore

_ALL_CAPABILITIES = frozenset(
    {
        MediaCapability.PLAY,
        MediaCapability.PAUSE,
        MediaCapability.RESUME,
        MediaCapability.STOP,
        MediaCapability.NEXT,
        MediaCapability.PREVIOUS,
        MediaCapability.SET_VOLUME,
    }
)


def _target(
    target_id: str,
    *,
    display_name: str,
    room: str | None = None,
    kind: TargetKind = TargetKind.SPEAKER,
    online: bool = True,
    capabilities: frozenset[MediaCapability] = _ALL_CAPABILITIES,
) -> AudioTarget:
    provider, native_id = target_id.split(":", 1)
    return AudioTarget(
        id=target_id,
        native_id=native_id,
        provider=provider,
        kind=kind,
        display_name=display_name,
        aliases=(),
        room=room,
        capabilities=capabilities,
        online=online,
        health="healthy" if online else "unavailable",
    )


class FakeAdapter:
    """In-memory provider double whose reads truthfully reflect applied state."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self._state: dict[str, MediaStateSnapshot] = {}
        self.calls: list[tuple[str, MediaAction, object]] = []
        self.reject_actions: set[MediaAction] = set()
        self.ignore_actions: set[MediaAction] = set()

    def seed(self, target_id: str, snapshot: MediaStateSnapshot) -> None:
        self._state[target_id] = snapshot

    def discover_targets(self) -> tuple[AudioTarget, ...]:
        return ()

    def execute_action(
        self,
        target: AudioTarget,
        action: MediaAction,
        *,
        value: object = None,
    ) -> MediaActionAcknowledgement:
        self.calls.append((target.id, action, value))
        if action in self.reject_actions:
            return MediaActionAcknowledgement(accepted=False, detail="provider rejected")
        if action not in self.ignore_actions:
            self._apply(target.id, action, value)
        return MediaActionAcknowledgement(accepted=True, detail="provider accepted")

    def get_state(self, target: AudioTarget) -> MediaStateSnapshot:
        return self._state.get(
            target.id,
            MediaStateSnapshot(
                target_id=target.id,
                playback=MediaState.UNKNOWN,
                observed_at=datetime.now(tz=UTC),
            ),
        )

    def _apply(self, target_id: str, action: MediaAction, value: object) -> None:
        current = self._state.get(
            target_id,
            MediaStateSnapshot(
                target_id=target_id,
                playback=MediaState.IDLE,
                observed_at=datetime.now(tz=UTC),
            ),
        )
        if action is MediaAction.PLAY:
            self._state[target_id] = MediaStateSnapshot(
                target_id=target_id,
                playback=MediaState.PLAYING,
                observed_at=datetime.now(tz=UTC),
                volume_percent=current.volume_percent,
                current_item_title=str(value) if value else current.current_item_title,
            )
        elif action is MediaAction.PAUSE:
            self._state[target_id] = MediaStateSnapshot(
                target_id=target_id,
                playback=MediaState.PAUSED,
                observed_at=datetime.now(tz=UTC),
                volume_percent=current.volume_percent,
                current_item_title=current.current_item_title,
            )
        elif action is MediaAction.RESUME:
            self._state[target_id] = MediaStateSnapshot(
                target_id=target_id,
                playback=MediaState.PLAYING,
                observed_at=datetime.now(tz=UTC),
                volume_percent=current.volume_percent,
                current_item_title=current.current_item_title,
            )
        elif action is MediaAction.STOP:
            self._state[target_id] = MediaStateSnapshot(
                target_id=target_id,
                playback=MediaState.STOPPED,
                observed_at=datetime.now(tz=UTC),
                volume_percent=current.volume_percent,
            )
        elif action is MediaAction.SET_VOLUME:
            self._state[target_id] = MediaStateSnapshot(
                target_id=target_id,
                playback=current.playback,
                observed_at=datetime.now(tz=UTC),
                volume_percent=(
                    float(value)
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    else current.volume_percent
                ),
                current_item_title=current.current_item_title,
            )
        else:
            self._state[target_id] = current


@pytest.fixture
def kitchen() -> AudioTarget:
    return _target("ha:media_player.kitchen", display_name="Kitchen Speaker", room="Kitchen")


@pytest.fixture
def living_room() -> AudioTarget:
    return _target(
        "ha:media_player.living_room", display_name="Living Room Speaker", room="Living Room"
    )


def _isolated_account_store() -> MediaAccountStore:
    return MediaAccountStore(root=Path(tempfile.mkdtemp(prefix="askrex-task4-media-")))


def _service(
    targets: list[AudioTarget],
    *,
    adapter: FakeAdapter,
    authorized_target_ids: dict[str, set[str]],
    account_store: MediaAccountStore | None = None,
    session_store: ActiveMediaSessionStore | None = None,
    clock=lambda: 1000.0,
    tmp_path=None,
) -> MediaService:
    registry = AudioTargetRegistry(targets, authorized_target_ids=authorized_target_ids)
    return MediaService(
        registry=registry,
        adapters={adapter.provider: adapter},
        account_store=account_store
        or (
            MediaAccountStore(root=tmp_path) if tmp_path is not None else _isolated_account_store()
        ),
        session_store=session_store or ActiveMediaSessionStore(ttl_seconds=300, clock=clock),
        clock=clock,
    )


def test_verified_pause_reports_verified_after_independent_state_match(kitchen) -> None:
    adapter = FakeAdapter("ha")
    adapter.seed(
        kitchen.id,
        MediaStateSnapshot(
            target_id=kitchen.id, playback=MediaState.PLAYING, observed_at=datetime.now(tz=UTC)
        ),
    )
    service = _service([kitchen], adapter=adapter, authorized_target_ids={"james": {kitchen.id}})

    result = service.execute(
        MediaCommand(action="pause", target_text=kitchen.id),
        user_id="james",
    )

    assert result.outcome == "verified"
    assert result.requested_target_id == kitchen.id
    assert result.mutation is not None
    assert result.mutation.outcome.value == "verified"
    assert adapter.calls == [(kitchen.id, MediaAction.PAUSE, None)]


def test_attempted_unverified_when_provider_accepts_but_state_does_not_match(kitchen) -> None:
    adapter = FakeAdapter("ha")
    adapter.ignore_actions = {MediaAction.PAUSE}
    adapter.seed(
        kitchen.id,
        MediaStateSnapshot(
            target_id=kitchen.id, playback=MediaState.PLAYING, observed_at=datetime.now(tz=UTC)
        ),
    )
    service = _service([kitchen], adapter=adapter, authorized_target_ids={"james": {kitchen.id}})

    result = service.execute(
        MediaCommand(action="pause", target_text=kitchen.id),
        user_id="james",
    )

    assert result.outcome == "attempted_unverified"
    assert result.requested_target_id == kitchen.id


def test_failed_when_provider_rejects_the_action(kitchen) -> None:
    adapter = FakeAdapter("ha")
    adapter.reject_actions = {MediaAction.STOP}
    service = _service([kitchen], adapter=adapter, authorized_target_ids={"james": {kitchen.id}})

    result = service.execute(
        MediaCommand(action="stop", target_text=kitchen.id),
        user_id="james",
    )

    assert result.outcome == "failed"
    assert result.mutation is None
    assert result.message == "provider rejected"


def test_ambiguous_target_never_dispatches_to_a_provider(kitchen, living_room) -> None:
    living_room_2 = _target(
        "sonos:living_room_2", display_name="Living Room Speaker", room="living room"
    )
    adapter = FakeAdapter("ha")
    service = _service(
        [kitchen, living_room, living_room_2],
        adapter=adapter,
        authorized_target_ids={"james": {kitchen.id, living_room.id, living_room_2.id}},
    )

    result = service.execute(
        MediaCommand(action="pause", target_text="living room"),
        user_id="james",
    )

    assert result.outcome == "ambiguous"
    assert result.ambiguous_ids
    assert adapter.calls == []


def test_not_authorized_target_is_reported_and_never_dispatched(kitchen) -> None:
    adapter = FakeAdapter("ha")
    service = _service([kitchen], adapter=adapter, authorized_target_ids={"james": set()})

    result = service.execute(
        MediaCommand(action="pause", target_text=kitchen.id),
        user_id="james",
    )

    assert result.outcome == "not_authorized"
    assert adapter.calls == []


def test_offline_target_is_reported_and_never_dispatched() -> None:
    offline = _target("ha:media_player.patio", display_name="Patio", online=False)
    adapter = FakeAdapter("ha")
    service = _service([offline], adapter=adapter, authorized_target_ids={"james": {offline.id}})

    result = service.execute(
        MediaCommand(action="pause", target_text=offline.id),
        user_id="james",
    )

    assert result.outcome == "offline"
    assert adapter.calls == []


def test_unsupported_capability_is_reported_without_dispatch(kitchen) -> None:
    limited = _target(
        "ha:media_player.limited",
        display_name="Limited Speaker",
        capabilities=frozenset({MediaCapability.PLAY}),
    )
    adapter = FakeAdapter("ha")
    service = _service([limited], adapter=adapter, authorized_target_ids={"james": {limited.id}})

    result = service.execute(
        MediaCommand(action="stop", target_text=limited.id),
        user_id="james",
    )

    assert result.outcome == "unsupported"
    assert result.requested_target_id == limited.id
    assert adapter.calls == []


@pytest.mark.parametrize("action", ["mute", "unmute"])
def test_mute_and_unmute_are_always_unsupported(kitchen, action: str) -> None:
    """No canonical MediaCapability represents mute; a truthful non-success is required."""
    adapter = FakeAdapter("ha")
    service = _service([kitchen], adapter=adapter, authorized_target_ids={"james": {kitchen.id}})

    result = service.execute(
        MediaCommand(action=action, target_text=kitchen.id),
        user_id="james",
    )

    assert result.outcome == "unsupported"
    assert adapter.calls == []


def test_state_query_reads_independently_and_never_touches_session(kitchen) -> None:
    adapter = FakeAdapter("ha")
    adapter.seed(
        kitchen.id,
        MediaStateSnapshot(
            target_id=kitchen.id,
            playback=MediaState.PLAYING,
            observed_at=datetime.now(tz=UTC),
            volume_percent=42.0,
        ),
    )
    session_store = ActiveMediaSessionStore(ttl_seconds=300, clock=lambda: 1000.0)
    service = _service(
        [kitchen],
        adapter=adapter,
        authorized_target_ids={"james": {kitchen.id}},
        session_store=session_store,
    )

    result = service.execute(
        MediaCommand(action="state", target_text=kitchen.id),
        user_id="james",
    )

    assert result.outcome == "read"
    assert result.state is not None
    assert result.state.playback == MediaState.PLAYING
    assert result.state.volume_percent == 42.0
    assert session_store.get("james") is None


def test_transfer_is_unsupported_even_with_active_session(kitchen, living_room) -> None:
    adapter = FakeAdapter("ha")
    session_store = ActiveMediaSessionStore(ttl_seconds=300, clock=lambda: 1000.0)
    session_store.set(
        ActiveMediaSession(
            user_id="james",
            target_id=kitchen.id,
            provider="ha",
            media_ref="track:42",
            updated_at=900.0,
        )
    )
    service = _service(
        [kitchen, living_room],
        adapter=adapter,
        authorized_target_ids={"james": {kitchen.id, living_room.id}},
        session_store=session_store,
        clock=lambda: 1000.0,
    )

    result = service.execute(
        MediaCommand(action="transfer", target_text=living_room.id),
        user_id="james",
    )

    assert result.outcome == "unsupported"
    assert adapter.calls == []
    unchanged = session_store.get("james")
    assert unchanged is not None
    assert unchanged.target_id == kitchen.id


def test_transfer_without_active_session_is_unsupported(kitchen, living_room) -> None:
    adapter = FakeAdapter("ha")
    service = _service(
        [kitchen, living_room],
        adapter=adapter,
        authorized_target_ids={"james": {kitchen.id, living_room.id}},
    )

    result = service.execute(
        MediaCommand(action="transfer", target_text=living_room.id),
        user_id="james",
    )

    assert result.outcome == "unsupported"
    assert adapter.calls == []


def test_session_is_not_updated_after_an_unverified_attempt(kitchen) -> None:
    adapter = FakeAdapter("ha")
    adapter.ignore_actions = {MediaAction.PLAY}
    session_store = ActiveMediaSessionStore(ttl_seconds=300, clock=lambda: 1000.0)
    service = _service(
        [kitchen],
        adapter=adapter,
        authorized_target_ids={"james": {kitchen.id}},
        session_store=session_store,
        clock=lambda: 1000.0,
    )

    result = service.execute(
        MediaCommand(action="play", query="jazz", target_text=kitchen.id),
        user_id="james",
    )

    assert result.outcome == "attempted_unverified"
    assert session_store.get("james") is None


def test_explicit_account_ref_must_belong_to_requesting_user(kitchen) -> None:
    account_store = _isolated_account_store()
    cole_account = account_store.put("cole", "ha", "main", generate_credential_ref(), "Cole HA")
    adapter = FakeAdapter("ha")
    service = _service(
        [kitchen],
        adapter=adapter,
        authorized_target_ids={"james": {kitchen.id}},
        account_store=account_store,
    )

    result = service.execute(
        MediaCommand(action="pause", target_text=kitchen.id),
        user_id="james",
        account_ref=cole_account,
    )

    assert result.outcome == "account_not_authorized"
    assert adapter.calls == []


def test_multiple_authorized_accounts_require_explicit_selection(kitchen) -> None:
    account_store = _isolated_account_store()
    account_store.put("james", "ha", "main", generate_credential_ref(), "James HA Main")
    account_store.put("james", "ha", "secondary", generate_credential_ref(), "James HA Secondary")
    adapter = FakeAdapter("ha")
    adapter.seed(
        kitchen.id,
        MediaStateSnapshot(
            target_id=kitchen.id, playback=MediaState.PLAYING, observed_at=datetime.now(tz=UTC)
        ),
    )
    service = _service(
        [kitchen],
        adapter=adapter,
        authorized_target_ids={"james": {kitchen.id}},
        account_store=account_store,
    )

    result = service.execute(
        MediaCommand(action="pause", target_text=kitchen.id),
        user_id="james",
    )

    assert result.outcome == "account_ambiguous"
    assert adapter.calls == []


def test_single_authorized_account_is_inferred_without_clarification(kitchen) -> None:
    account_store = _isolated_account_store()
    account_store.put("james", "ha", "main", generate_credential_ref(), "James HA Main")
    adapter = FakeAdapter("ha")
    adapter.seed(
        kitchen.id,
        MediaStateSnapshot(
            target_id=kitchen.id, playback=MediaState.PLAYING, observed_at=datetime.now(tz=UTC)
        ),
    )
    service = _service(
        [kitchen],
        adapter=adapter,
        authorized_target_ids={"james": {kitchen.id}},
        account_store=account_store,
    )

    result = service.execute(
        MediaCommand(action="pause", target_text=kitchen.id),
        user_id="james",
    )

    assert result.outcome == "verified"
    assert adapter.calls == [(kitchen.id, MediaAction.PAUSE, None)]


def test_origin_device_is_used_when_no_target_text_is_supplied(kitchen) -> None:
    adapter = FakeAdapter("ha")
    adapter.seed(
        kitchen.id,
        MediaStateSnapshot(
            target_id=kitchen.id, playback=MediaState.PLAYING, observed_at=datetime.now(tz=UTC)
        ),
    )
    registry = AudioTargetRegistry(
        [kitchen],
        authorized_target_ids={"james": {kitchen.id}},
        origin_device_targets={"mic_kitchen": kitchen.id},
    )
    service = MediaService(
        registry=registry,
        adapters={"ha": adapter},
        account_store=_isolated_account_store(),
        session_store=ActiveMediaSessionStore(ttl_seconds=300, clock=lambda: 1000.0),
        clock=lambda: 1000.0,
    )

    result = service.execute(
        MediaCommand(action="pause"),
        user_id="james",
        origin_device_id="mic_kitchen",
    )

    assert result.outcome == "verified"
    assert result.requested_target_id == kitchen.id


def test_reverify_independently_confirms_current_provider_state(kitchen) -> None:
    adapter = FakeAdapter("ha")
    adapter.seed(
        kitchen.id,
        MediaStateSnapshot(
            target_id=kitchen.id, playback=MediaState.PAUSED, observed_at=datetime.now(tz=UTC)
        ),
    )
    service = _service([kitchen], adapter=adapter, authorized_target_ids={"james": {kitchen.id}})

    assert service.reverify(
        target_id=kitchen.id,
        provider="ha",
        user_id="james",
        expected={"playback": ["paused"]},
    )
    assert not service.reverify(
        target_id=kitchen.id,
        provider="ha",
        user_id="james",
        expected={"playback": ["playing"]},
    )


def test_reverify_fails_closed_for_unauthorized_or_unknown_targets(kitchen) -> None:
    adapter = FakeAdapter("ha")
    service = _service([kitchen], adapter=adapter, authorized_target_ids={"james": set()})

    assert not service.reverify(
        target_id=kitchen.id,
        provider="ha",
        user_id="james",
        expected={"playback": ["paused"]},
    )
    assert not service.reverify(
        target_id="ha:does-not-exist",
        provider="ha",
        user_id="james",
        expected={"playback": ["paused"]},
    )


def test_wrong_target_snapshot_cannot_verify_mutation(kitchen) -> None:
    adapter = FakeAdapter("ha")
    adapter.ignore_actions = {MediaAction.PAUSE}
    adapter.seed(
        kitchen.id,
        MediaStateSnapshot(
            target_id="ha:media_player.other",
            playback=MediaState.PAUSED,
            observed_at=datetime.now(tz=UTC),
        ),
    )
    service = _service([kitchen], adapter=adapter, authorized_target_ids={"james": {kitchen.id}})

    result = service.execute(MediaCommand(action="pause", target_text=kitchen.id), user_id="james")

    assert result.outcome == "attempted_unverified"
    assert not service.reverify(
        target_id=kitchen.id,
        provider="ha",
        user_id="james",
        expected={"playback": ["paused"]},
    )


def test_playing_old_content_does_not_verify_requested_play(kitchen) -> None:
    adapter = FakeAdapter("ha")
    adapter.ignore_actions = {MediaAction.PLAY}
    adapter.seed(
        kitchen.id,
        MediaStateSnapshot(
            target_id=kitchen.id,
            playback=MediaState.PLAYING,
            observed_at=datetime.now(tz=UTC),
            current_item_title="old song",
        ),
    )
    service = _service([kitchen], adapter=adapter, authorized_target_ids={"james": {kitchen.id}})

    result = service.execute(
        MediaCommand(action="play", query="jazz", target_text=kitchen.id),
        user_id="james",
    )

    assert result.outcome == "attempted_unverified"


def test_execute_refreshes_registry_before_target_resolution() -> None:
    stale = _target("ha:media_player.old", display_name="Old Speaker")
    fresh = _target("ha:media_player.den", display_name="Den Speaker", room="Den")
    stale_registry = AudioTargetRegistry([stale], authorized_target_ids={"james": {stale.id}})
    fresh_registry = AudioTargetRegistry([fresh], authorized_target_ids={"james": {fresh.id}})
    adapter = FakeAdapter("ha")

    class RefreshingService(MediaService):
        refresh_calls = 0

        def _refresh_registry(self) -> None:
            self.refresh_calls += 1
            self._registry = fresh_registry

    service = RefreshingService(registry=stale_registry, adapters={"ha": adapter})
    result = service.execute(MediaCommand(action="state", target_text=fresh.id), user_id="james")

    assert service.refresh_calls == 1
    assert result.outcome == "read"
    assert result.requested_target_id == fresh.id


@pytest.mark.parametrize(("command_text", "expected_muted"), [("mute", True), ("unmute", False)])
def test_mute_commands_use_verified_canonical_lifecycle(
    command_text: str, expected_muted: bool
) -> None:
    target = _target(
        "ha:media_player.den",
        display_name="Den",
        capabilities=frozenset({"mute"}),  # type: ignore[arg-type]
    )

    class MutingAdapter:
        provider = "ha"
        muted = not expected_muted

        def execute_action(self, target, action, *, value=None):
            self.muted = str(action) == "mute"
            return MediaActionAcknowledgement(accepted=True, detail="accepted")

        def get_state(self, target):
            return MediaStateSnapshot(
                target_id=target.id,
                playback=MediaState.PLAYING,
                observed_at=datetime.now(tz=UTC),
                muted=self.muted,
            )

    registry = AudioTargetRegistry([target], authorized_target_ids={"james": {target.id}})
    service = MediaService(registry=registry, adapters={"ha": MutingAdapter()})
    result = service.execute(
        MediaCommand(action=command_text, target_text=target.id), user_id="james"
    )

    assert result.outcome == "verified"
    assert result.mutation is not None
    assert getattr(result.mutation.observed_state, "muted", None) is expected_muted


def test_verified_media_session_publishes_active_context(kitchen) -> None:
    def clock() -> float:
        return 1000.0

    active = ActiveContextStore(clock=clock)
    sessions = ActiveMediaSessionStore(
        ttl_seconds=300,
        clock=clock,
        active_context_store=active,
    )
    adapter = FakeAdapter("ha")
    adapter.seed(
        kitchen.id,
        MediaStateSnapshot(
            target_id=kitchen.id,
            playback=MediaState.IDLE,
            observed_at=datetime.now(tz=UTC),
        ),
    )
    service = _service(
        [kitchen],
        adapter=adapter,
        authorized_target_ids={"james": {kitchen.id}},
        session_store=sessions,
        clock=clock,
    )

    result = service.execute(
        MediaCommand(action="play", query="Miles Davis", target_text=kitchen.id),
        user_id="james",
    )

    assert result.outcome == "verified"
    ref = active.get("james", "media", kitchen.id)
    assert ref is not None
    assert ref.payload["target_id"] == kitchen.id
    assert ref.payload["provider"] == "ha"
    assert str(ref.payload["media_ref"]).startswith("query:")
    assert ref.expires_at == 1300.0
