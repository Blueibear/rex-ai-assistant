from __future__ import annotations

from datetime import UTC, datetime

from rex.credential_vault import generate_credential_ref
from rex.media.accounts import MediaAccountRef
from rex.media.models import AudioTarget, MediaCapability, TargetKind
from rex.media.registry import AudioTargetRegistry
from rex.media.service import MediaServiceResult
from rex.output_routing.models import OutputKind, ResolvedRoute
from rex.runtime.turn import IdentityResolution


def _registry() -> AudioTargetRegistry:
    target = AudioTarget(
        id="ha:media_player.kitchen",
        native_id="media_player.kitchen",
        provider="ha",
        kind=TargetKind.SPEAKER,
        display_name="Kitchen",
        aliases=(),
        room="Kitchen",
        capabilities=frozenset({MediaCapability.PLAY, MediaCapability.SET_VOLUME}),
        online=True,
        health="healthy",
    )
    return AudioTargetRegistry(
        (target,),
        authorized_target_ids={"james": {target.id}},
        origin_device_targets={"mic_kitchen": target.id},
    )


def test_media_tool_passes_trusted_account_and_route_volume_to_service(monkeypatch) -> None:
    import rex.media.tools as tools

    account = MediaAccountRef(
        user_id="james",
        provider="ha",
        account_id="main",
        credential_ref=generate_credential_ref(),
        display_name="James Home Audio",
    )

    class Routing:
        def __init__(self) -> None:
            self.account_call = None

        def resolve(self, **kwargs):
            assert kwargs["user_id"] == "james"
            assert kwargs["output_kind"] is OutputKind.MEDIA
            return ResolvedRoute(
                output_kind=OutputKind.MEDIA,
                target_id="ha:media_player.kitchen",
                reason="request_origin",
                target_volume=35,
            )

        def resolve_media_account(self, **kwargs):
            self.account_call = kwargs
            return account

    routing = Routing()
    observed: dict[str, object] = {}

    class FakeMediaService:
        _registry = _registry()

        def execute(self, command, **kwargs):
            observed["command"] = command
            observed.update(kwargs)
            return MediaServiceResult(
                outcome="attempted_unverified",
                requested_target_id="ha:media_player.kitchen",
            )

    monkeypatch.setattr(tools, "get_output_routing_service", lambda: routing)
    monkeypatch.setattr(
        tools,
        "user_local_now",
        lambda _user_id: datetime(2026, 8, 21, 12, 0, tzinfo=UTC),
    )
    monkeypatch.setattr(tools, "_media_service", FakeMediaService())

    result = tools.media_manage(
        action="play",
        query="Miles Davis",
        origin_device_id="mic_kitchen",
        _user_id="james",
    )

    assert result["status"] == "attempted_unverified"
    assert observed["account_ref"] == account
    assert observed["route_volume"] == 35
    assert routing.account_call == {
        "active_user_id": "james",
        "identity_resolution": IdentityResolution.EXPLICIT,
        "requested_account_id": None,
        "operation": "play",
    }
