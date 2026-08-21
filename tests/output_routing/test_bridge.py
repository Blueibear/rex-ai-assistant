from __future__ import annotations

from bridge.rex_output_routing_bridge import handle_output_routing_request
from rex.credential_vault import generate_credential_ref
from rex.media.accounts import MediaAccountStore
from rex.media.models import AudioTarget, MediaCapability, TargetKind
from rex.media.registry import AudioTargetRegistry
from rex.output_routing.models import UserOutputPolicy
from rex.output_routing.service import OutputRoutingService


def _target(target_id: str, name: str, *, online: bool = True) -> AudioTarget:
    return AudioTarget(
        id=target_id,
        native_id=target_id.split(":", 1)[-1],
        provider=target_id.split(":", 1)[0],
        kind=TargetKind.SPEAKER,
        display_name=name,
        aliases=(),
        room=None,
        capabilities=frozenset({MediaCapability.PLAY}),
        online=online,
        health="healthy" if online else "offline",
    )


def _services(tmp_path):
    kitchen = _target("ha:media_player.kitchen", "Kitchen")
    den = _target("ha:media_player.den", "Den", online=False)
    registry = AudioTargetRegistry(
        (kitchen, den),
        authorized_target_ids={"james": {kitchen.id, den.id}, "cole": {kitchen.id}},
    )
    routing = OutputRoutingService(registry, root=tmp_path / "routing")
    accounts = MediaAccountStore(tmp_path / "accounts")
    return registry, routing, accounts


def test_bridge_reads_and_updates_only_bound_users_policy(tmp_path) -> None:
    registry, routing, accounts = _services(tmp_path)

    body, code = handle_output_routing_request(
        {
            "command": "update_policy",
            "policy": {
                "spoken_response_target_id": "ha:media_player.kitchen",
                "timer_target_id": None,
                "alarm_target_id": "ha:media_player.den",
                "media_target_id": "ha:media_player.kitchen",
            },
        },
        registry=registry,
        routing=routing,
        media_accounts=accounts,
        bound_user_id="james",
    )

    assert code == 0
    assert body["ok"] is True
    assert body["policy"]["alarm_target_id"] == "ha:media_player.den"
    assert routing.get_policy("james").media_target_id == "ha:media_player.kitchen"
    assert routing.get_policy("cole") == UserOutputPolicy()


def test_bridge_rejects_cross_user_override(tmp_path) -> None:
    registry, routing, accounts = _services(tmp_path)

    body, code = handle_output_routing_request(
        {"command": "get_policy", "user_id": "cole"},
        registry=registry,
        routing=routing,
        media_accounts=accounts,
        bound_user_id="james",
    )

    assert code == 1
    assert body["ok"] is False


def test_bridge_media_account_list_never_exposes_credential_reference(tmp_path) -> None:
    registry, routing, accounts = _services(tmp_path)
    accounts.put(
        "james",
        "apple_music",
        "main",
        generate_credential_ref(),
        "James Apple Music",
    )

    body, code = handle_output_routing_request(
        {"command": "list_media_accounts"},
        registry=registry,
        routing=routing,
        media_accounts=accounts,
        bound_user_id="james",
    )

    assert code == 0
    assert body == {
        "ok": True,
        "accounts": [
            {
                "provider": "apple_music",
                "account_id": "main",
                "display_name": "James Apple Music",
            }
        ],
    }
    assert "credential" not in repr(body).lower()


def test_bridge_default_media_account_must_belong_to_bound_user(tmp_path) -> None:
    registry, routing, accounts = _services(tmp_path)
    accounts.put(
        "cole",
        "apple_music",
        "main",
        generate_credential_ref(),
        "Cole Apple Music",
    )

    body, code = handle_output_routing_request(
        {
            "command": "set_default_media_account",
            "provider": "apple_music",
            "account_id": "main",
        },
        registry=registry,
        routing=routing,
        media_accounts=accounts,
        bound_user_id="james",
    )

    assert code == 1
    assert body["ok"] is False
    assert routing.get_policy("james").default_media_account_id is None
