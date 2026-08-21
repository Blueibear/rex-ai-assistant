from __future__ import annotations

from datetime import UTC, datetime

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


class VolumeAwareAdapter:
    provider = "ha"

    def __init__(self, target_id: str, *, volume: float = 20.0) -> None:
        self.target_id = target_id
        self.volume = volume
        self.playback = MediaState.PAUSED
        self.calls: list[tuple[MediaAction, object]] = []
        self.ignore_volume_updates = False

    def discover_targets(self):
        return ()

    def execute_action(self, target, action, *, value=None):
        self.calls.append((action, value))
        if action is MediaAction.SET_VOLUME and not self.ignore_volume_updates:
            self.volume = float(value)
        elif action is MediaAction.PLAY:
            self.playback = MediaState.PLAYING
        elif action is MediaAction.PAUSE:
            self.playback = MediaState.PAUSED
        return MediaActionAcknowledgement(accepted=True)

    def get_state(self, target):
        return MediaStateSnapshot(
            target_id=self.target_id,
            playback=self.playback,
            observed_at=datetime.now(tz=UTC),
            volume_percent=self.volume,
        )


def _target() -> AudioTarget:
    return AudioTarget(
        id="ha:media_player.kitchen",
        native_id="media_player.kitchen",
        provider="ha",
        kind=TargetKind.SPEAKER,
        display_name="Kitchen",
        aliases=(),
        room="Kitchen",
        capabilities=frozenset(
            {MediaCapability.PLAY, MediaCapability.PAUSE, MediaCapability.SET_VOLUME}
        ),
        online=True,
        health="healthy",
    )


def _service(tmp_path, adapter: VolumeAwareAdapter, accounts: MediaAccountStore) -> MediaService:
    target = _target()
    return MediaService(
        registry=AudioTargetRegistry(
            (target,),
            authorized_target_ids={"james": {target.id}},
        ),
        adapters={adapter.provider: adapter},
        account_store=accounts,
    )


def test_media_account_ownership_is_separate_from_output_provider(tmp_path) -> None:
    accounts = MediaAccountStore(tmp_path / "accounts")
    apple = accounts.put(
        "james",
        "apple_music",
        "main",
        generate_credential_ref(),
        "James Apple Music",
    )
    adapter = VolumeAwareAdapter(_target().id)
    service = _service(tmp_path, adapter, accounts)

    result = service.execute(
        MediaCommand(action="pause", target_text=_target().id),
        user_id="james",
        account_ref=apple,
    )

    assert result.outcome == "verified"
    assert adapter.calls == [(MediaAction.PAUSE, None)]


def test_routed_volume_is_temporary_and_restored_after_verified_media_action(tmp_path) -> None:
    accounts = MediaAccountStore(tmp_path / "accounts")
    adapter = VolumeAwareAdapter(_target().id, volume=20.0)
    service = _service(tmp_path, adapter, accounts)

    result = service.execute(
        MediaCommand(action="play", query="jazz", target_text=_target().id),
        user_id="james",
        route_volume=35,
    )

    assert result.outcome == "verified"
    assert adapter.calls == [
        (MediaAction.SET_VOLUME, 35),
        (MediaAction.PLAY, "jazz"),
        (MediaAction.SET_VOLUME, 20.0),
    ]
    assert adapter.volume == 20.0
    assert result.state is not None
    assert result.state.volume_percent == 20.0
    assert result.state.playback is MediaState.PLAYING


def test_unverified_temporary_volume_prevents_media_dispatch_and_restores_if_possible(tmp_path) -> None:
    accounts = MediaAccountStore(tmp_path / "accounts")
    adapter = VolumeAwareAdapter(_target().id, volume=20.0)
    adapter.ignore_volume_updates = True
    service = _service(tmp_path, adapter, accounts)

    result = service.execute(
        MediaCommand(action="play", query="jazz", target_text=_target().id),
        user_id="james",
        route_volume=35,
    )

    assert result.outcome == "failed"
    assert result.message == "temporary route volume could not be verified"
    assert adapter.calls == [
        (MediaAction.SET_VOLUME, 35),
        (MediaAction.SET_VOLUME, 20.0),
    ]
    assert all(action is not MediaAction.PLAY for action, _value in adapter.calls)
