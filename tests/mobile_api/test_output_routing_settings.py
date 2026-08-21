from __future__ import annotations

from rex.credential_vault import generate_credential_ref
from rex.media.accounts import MediaAccountStore
from rex.media.models import AudioTarget, MediaCapability, TargetKind
from rex.media.registry import AudioTargetRegistry
from rex.output_routing.service import OutputRoutingService
from tests.mobile_api.conftest import auth_header, create_user, paired_login_tokens


def _backend(tmp_path):
    target = AudioTarget(
        id="ha:media_player.kitchen",
        native_id="media_player.kitchen",
        provider="ha",
        kind=TargetKind.SPEAKER,
        display_name="Kitchen",
        aliases=(),
        room="Kitchen",
        capabilities=frozenset({MediaCapability.PLAY}),
        online=True,
        health="healthy",
    )
    registry = AudioTargetRegistry(
        (target,),
        authorized_target_ids={"james": {target.id}, "cole": {target.id}},
    )
    routing = OutputRoutingService(registry, root=tmp_path / "routing")
    accounts = MediaAccountStore(tmp_path / "accounts")
    accounts.put(
        "james",
        "apple_music",
        "main",
        generate_credential_ref(),
        "James Apple Music",
    )
    return registry, routing, accounts


def test_mobile_output_policy_is_user_bound_and_shared_with_backend(
    client, monkeypatch, tmp_path
) -> None:
    create_user("james", "pw-123456")
    tokens = paired_login_tokens(
        client,
        "james",
        "pw-123456",
        scopes=["settings.read", "settings.write"],
    )
    backend = _backend(tmp_path)
    monkeypatch.setattr(
        "rex.mobile_api.routes.settings._build_routing_backend",
        lambda: backend,
    )

    response = client.put(
        "/mobile/settings/output-routing",
        headers=auth_header(tokens["access_token"]),
        json={"media_target_id": "ha:media_player.kitchen"},
    )
    assert response.status_code == 200, response.get_json()
    assert response.get_json()["policy"]["media_target_id"] == "ha:media_player.kitchen"
    assert backend[1].get_policy("james").media_target_id == "ha:media_player.kitchen"

    loaded = client.get(
        "/mobile/settings/output-routing",
        headers=auth_header(tokens["access_token"]),
    )
    assert loaded.status_code == 200
    assert loaded.get_json()["policy"]["media_target_id"] == "ha:media_player.kitchen"


def test_mobile_cannot_write_another_users_routing_policy(client, monkeypatch, tmp_path) -> None:
    create_user("james", "pw-123456")
    tokens = paired_login_tokens(
        client,
        "james",
        "pw-123456",
        scopes=["settings.read", "settings.write"],
    )
    backend = _backend(tmp_path)
    monkeypatch.setattr(
        "rex.mobile_api.routes.settings._build_routing_backend",
        lambda: backend,
    )

    response = client.put(
        "/mobile/settings/output-routing",
        headers=auth_header(tokens["access_token"]),
        json={"user_id": "cole", "media_target_id": "ha:media_player.kitchen"},
    )
    assert response.status_code == 403
    assert backend[1].get_policy("james").media_target_id is None
    assert backend[1].get_policy("cole").media_target_id is None


def test_mobile_settings_write_scope_is_required(client, monkeypatch, tmp_path) -> None:
    create_user("james", "pw-123456")
    tokens = paired_login_tokens(
        client,
        "james",
        "pw-123456",
        scopes=["settings.read"],
    )
    backend = _backend(tmp_path)
    monkeypatch.setattr(
        "rex.mobile_api.routes.settings._build_routing_backend",
        lambda: backend,
    )

    response = client.put(
        "/mobile/settings/output-routing",
        headers=auth_header(tokens["access_token"]),
        json={"media_target_id": "ha:media_player.kitchen"},
    )
    assert response.status_code == 403


def test_mobile_media_accounts_expose_safe_metadata_only(client, monkeypatch, tmp_path) -> None:
    create_user("james", "pw-123456")
    tokens = paired_login_tokens(
        client,
        "james",
        "pw-123456",
        scopes=["settings.read"],
    )
    backend = _backend(tmp_path)
    monkeypatch.setattr(
        "rex.mobile_api.routes.settings._build_routing_backend",
        lambda: backend,
    )

    response = client.get(
        "/mobile/settings/output-routing/accounts",
        headers=auth_header(tokens["access_token"]),
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["accounts"] == [
        {
            "provider": "apple_music",
            "account_id": "main",
            "display_name": "James Apple Music",
        }
    ]
    assert "credential" not in repr(body).lower()
