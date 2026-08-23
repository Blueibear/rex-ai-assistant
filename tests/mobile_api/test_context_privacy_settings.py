from __future__ import annotations

from rex.context.location_policy import LocationGrantStore
from rex.context.source_policy import ContextSourcePolicyStore, ContextSourceType
from rex.knowledge_base import KnowledgeBase
from tests.mobile_api.conftest import auth_header, create_user, paired_login_tokens


def _backend(tmp_path):
    from rex.context.privacy import ContextPrivacyPreferenceStore, ContextPrivacyService

    policy = ContextSourcePolicyStore(tmp_path / "policy")
    location = LocationGrantStore(tmp_path / "location", source_policy_store=policy)
    knowledge = KnowledgeBase(
        tmp_path / "docs.json",
        tmp_path / "index.json",
        source_policy_store=policy,
    )
    preferences = ContextPrivacyPreferenceStore(tmp_path / "preferences")
    return ContextPrivacyService(
        source_policy_store=policy,
        location_grant_store=location,
        knowledge_base=knowledge,
        preference_store=preferences,
    )


def test_mobile_owner_can_toggle_location_assistance(client, monkeypatch, tmp_path):
    user_id = create_user("james", "pw-123456")
    tokens = paired_login_tokens(
        client,
        "james",
        "pw-123456",
        scopes=["settings.read", "settings.write"],
    )
    backend = _backend(tmp_path)
    monkeypatch.setattr(
        "rex.mobile_api.routes.settings._build_context_privacy_service",
        lambda: backend,
    )

    response = client.put(
        "/mobile/settings/context/location",
        headers=auth_header(tokens["access_token"]),
        json={"location_assist": True},
    )

    assert response.status_code == 200, response.get_json()
    assert response.get_json()["location"]["location_assist"] is True
    assert backend.location_grant_store.get(user_id).location_assist is True


def test_mobile_admin_cannot_change_another_users_location_assist(client, monkeypatch, tmp_path):
    create_user("james", "pw-123456", admin=True)
    cole_id = create_user("cole", "pw-123456")
    tokens = paired_login_tokens(
        client,
        "james",
        "pw-123456",
        scopes=["settings.read", "settings.write"],
    )
    backend = _backend(tmp_path)
    monkeypatch.setattr(
        "rex.mobile_api.routes.settings._build_context_privacy_service",
        lambda: backend,
    )

    response = client.put(
        "/mobile/settings/context/location",
        headers=auth_header(tokens["access_token"]),
        json={"user_id": cole_id, "location_assist": True},
    )

    assert response.status_code == 403
    assert backend.location_grant_store.get(cole_id).location_assist is False


def test_mobile_context_settings_read_and_proactive_write(client, monkeypatch, tmp_path):
    user_id = create_user("james", "pw-123456")
    tokens = paired_login_tokens(
        client,
        "james",
        "pw-123456",
        scopes=["settings.read", "settings.write"],
    )
    backend = _backend(tmp_path)
    monkeypatch.setattr(
        "rex.mobile_api.routes.settings._build_context_privacy_service",
        lambda: backend,
    )

    updated = client.put(
        "/mobile/settings/context/proactive",
        headers=auth_header(tokens["access_token"]),
        json={"enabled": False},
    )
    loaded = client.get(
        "/mobile/settings/context",
        headers=auth_header(tokens["access_token"]),
    )

    assert updated.status_code == 200
    assert updated.get_json()["proactive_assistance"] is False
    assert loaded.status_code == 200
    assert loaded.get_json()["proactive_assistance"] is False
    assert backend.preference_store.get(user_id).proactive_assistance is False


def test_mobile_can_update_owned_source_and_upload_policy(client, monkeypatch, tmp_path):
    user_id = create_user("james", "pw-123456")
    tokens = paired_login_tokens(
        client,
        "james",
        "pw-123456",
        scopes=["settings.read", "settings.write"],
    )
    backend = _backend(tmp_path)
    backend.source_policy_store.register_source(
        "integration:calendar",
        ContextSourceType.INTEGRATION,
        owner_user_id=user_id,
    )
    doc = backend.knowledge_base.ingest_text(
        "private notes",
        "Trip",
        owner_user_id=user_id,
        audience_scope="private",
        context_enabled=False,
    )
    monkeypatch.setattr(
        "rex.mobile_api.routes.settings._build_context_privacy_service",
        lambda: backend,
    )

    source_response = client.put(
        "/mobile/settings/context/source",
        headers=auth_header(tokens["access_token"]),
        json={"source_id": "integration:calendar", "enabled": False},
    )
    upload_response = client.put(
        "/mobile/settings/context/upload",
        headers=auth_header(tokens["access_token"]),
        json={
            "doc_id": doc.doc_id,
            "audience_scope": "household",
            "context_enabled": True,
        },
    )

    assert source_response.status_code == 200
    assert source_response.get_json()["source"]["context_enabled"] is False
    assert upload_response.status_code == 200
    assert upload_response.get_json()["upload"]["audience_scope"] == "household"
    saved = backend.knowledge_base.get_document_for_user(
        doc.doc_id,
        requester_user_id=user_id,
    )
    assert saved is not None and saved.context_enabled is True


def test_mobile_location_share_is_owner_controlled(client, monkeypatch, tmp_path):
    user_id = create_user("james", "pw-123456")
    cole_id = create_user("cole", "pw-123456")
    tokens = paired_login_tokens(
        client,
        "james",
        "pw-123456",
        scopes=["settings.read", "settings.write"],
    )
    backend = _backend(tmp_path)
    monkeypatch.setattr(
        "rex.mobile_api.routes.settings._build_context_privacy_service",
        lambda: backend,
    )

    response = client.put(
        "/mobile/settings/context/location-share",
        headers=auth_header(tokens["access_token"]),
        json={"recipient_user_id": cole_id, "enabled": True},
    )

    assert response.status_code == 200
    assert cole_id in response.get_json()["location"]["shared_with"]
    assert backend.location_grant_store.can_share(user_id, cole_id)


def test_mobile_privacy_mutations_require_settings_write(client, monkeypatch, tmp_path):
    create_user("james", "pw-123456")
    tokens = paired_login_tokens(
        client,
        "james",
        "pw-123456",
        scopes=["settings.read"],
    )
    backend = _backend(tmp_path)
    monkeypatch.setattr(
        "rex.mobile_api.routes.settings._build_context_privacy_service",
        lambda: backend,
    )

    response = client.put(
        "/mobile/settings/context/proactive",
        headers=auth_header(tokens["access_token"]),
        json={"enabled": False},
    )

    assert response.status_code == 403
