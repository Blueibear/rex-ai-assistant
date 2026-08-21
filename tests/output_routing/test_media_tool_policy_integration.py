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
from rex.media.registry import AudioTargetRegistry
from rex.media.service import MediaService
from rex.media.tools import media_manage, set_media_service
from rex.output_routing.models import OutputKind, ResolvedRoute
from rex.runtime.invocation import turn_invocation
from rex.runtime.turn import IdentityResolution, TurnSource


class RecordingAdapter:
    provider = "ha"

    def __init__(self, target_id: str) -> None:
        self.target_id = target_id
        self.playback = MediaState.PLAYING
        self.volume = 20.0
        self.calls: list[tuple[MediaAction, object]] = []

    def discover_targets(self):
        return ()

    def execute_action(self, target, action, *, value=None):
        self.calls.append((action, value))
        if action is MediaAction.SET_VOLUME:
            self.volume = float(value)
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


class RecordingRouting:
    def __init__(self, target_id, account) -> None:
        self.target_id = target_id
        self.account = account
        self.route_calls: list[dict[str, object]] = []
        self.account_calls: list[dict[str, object]] = []

    def resolve(self, **kwargs):
        self.route_calls.append(kwargs)
        return ResolvedRoute(
            output_kind=OutputKind.MEDIA,
            target_id=self.target_id,
            reason="request_origin",
            target_volume=35,
        )

    def resolve_media_account(self, **kwargs):
        self.account_calls.append(kwargs)
        return self.account


def _target() -> AudioTarget:
    return AudioTarget(
        id="ha:media_player.kitchen",
        native_id="media_player.kitchen",
        provider="ha",
        kind=TargetKind.SPEAKER,
        display_name="Kitchen",
        aliases=(),
        room="Kitchen",
        capabilities=frozenset({MediaCapability.PAUSE, MediaCapability.SET_VOLUME}),
        online=True,
        health="healthy",
    )


def test_media_tool_carries_trusted_identity_account_and_route_volume(monkeypatch, tmp_path) -> None:
    target = _target()
    accounts = MediaAccountStore(tmp_path / "accounts")
    apple = accounts.put(
        "james",
        "apple_music",
        "main",
        generate_credential_ref(),
        "James Apple Music",
    )
    adapter = RecordingAdapter(target.id)
    service = MediaService(
        registry=AudioTargetRegistry(
            (target,),
            authorized_target_ids={"james": {target.id}},
            origin_device_targets={"mic_kitchen": target.id},
        ),
        adapters={"ha": adapter},
        account_store=accounts,
    )
    routing = RecordingRouting(target.id, apple)
    monkeypatch.setattr("rex.media.tools.get_output_routing_service", lambda: routing)
    monkeypatch.setattr(
        "rex.media.tools.user_local_now",
        lambda _user_id: datetime(2026, 8, 21, 7, 0, tzinfo=UTC),
    )
    set_media_service(service)
    try:
        with turn_invocation(
            TurnSource.VOICE,
            device_id="mic_kitchen",
            identity_resolution=IdentityResolution.VOICE_RECOGNIZED,
        ):
            output = media_manage(action="pause", _user_id="james")
    finally:
        set_media_service(None)

    assert output["lifecycle_state"] == "verified"
    assert adapter.calls == [
        (MediaAction.SET_VOLUME, 35),
        (MediaAction.PAUSE, None),
        (MediaAction.SET_VOLUME, 20.0),
    ]
    assert routing.route_calls[0]["origin_device_id"] == "mic_kitchen"
    assert routing.account_calls == [
        {
            "active_user_id": "james",
            "identity_resolution": IdentityResolution.VOICE_RECOGNIZED,
            "requested_account_id": None,
            "operation": "pause",
        }
    ]
