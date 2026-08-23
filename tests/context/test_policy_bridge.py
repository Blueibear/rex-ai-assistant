from __future__ import annotations

import pytest

from rex.context.location_policy import LocationGrantStore
from rex.context.source_policy import ContextSourcePolicyStore, ContextSourceType
from rex.knowledge_base import KnowledgeBase


def _service(tmp_path):
    from rex.context.privacy import ContextPrivacyPreferenceStore, ContextPrivacyService

    policy = ContextSourcePolicyStore(tmp_path / "policy")
    location = LocationGrantStore(tmp_path / "location", source_policy_store=policy)
    knowledge = KnowledgeBase(
        tmp_path / "docs.json",
        tmp_path / "index.json",
        source_policy_store=policy,
    )
    preferences = ContextPrivacyPreferenceStore(tmp_path / "preferences")
    service = ContextPrivacyService(
        source_policy_store=policy,
        location_grant_store=location,
        knowledge_base=knowledge,
        preference_store=preferences,
    )
    return service, policy, location, knowledge, preferences


def test_admin_shaped_caller_cannot_enable_another_users_location(tmp_path):
    from bridge.rex_context_policy_bridge import handle_context_policy_request

    service, _policy, location, _knowledge, _preferences = _service(tmp_path)
    body, code = handle_context_policy_request(
        {
            "command": "set_location_assist",
            "user_id": "cole",
            "enabled": True,
            "caller_permissions": ["admin"],
        },
        service=service,
        bound_user_id="james",
    )

    assert code != 0
    assert body["ok"] is False
    assert location.get("cole").location_assist is False


def test_owner_can_promote_owned_upload_to_household(tmp_path):
    from bridge.rex_context_policy_bridge import handle_context_policy_request

    service, _policy, _location, knowledge, _preferences = _service(tmp_path)
    doc = knowledge.ingest_text(
        "private notes",
        "Trip",
        owner_user_id="james",
        audience_scope="private",
        context_enabled=False,
    )

    body, code = handle_context_policy_request(
        {
            "command": "update_upload_policy",
            "doc_id": doc.doc_id,
            "audience_scope": "household",
            "context_enabled": True,
        },
        service=service,
        bound_user_id="james",
    )

    assert code == 0 and body["ok"] is True
    saved = knowledge.get_document_for_user(doc.doc_id, requester_user_id="james")
    assert saved is not None
    assert saved.audience_scope == "household"
    assert saved.context_enabled is True


def test_other_user_cannot_change_private_upload_policy(tmp_path):
    from bridge.rex_context_policy_bridge import handle_context_policy_request

    service, _policy, _location, knowledge, _preferences = _service(tmp_path)
    doc = knowledge.ingest_text(
        "private notes",
        "Trip",
        owner_user_id="james",
        audience_scope="private",
        context_enabled=True,
    )

    body, code = handle_context_policy_request(
        {
            "command": "update_upload_policy",
            "doc_id": doc.doc_id,
            "audience_scope": "household",
            "context_enabled": True,
        },
        service=service,
        bound_user_id="cole",
    )

    assert code != 0 and body["ok"] is False
    saved = knowledge.get_document_for_user(doc.doc_id, requester_user_id="james")
    assert saved is not None and saved.audience_scope == "private"


def test_owner_can_toggle_connected_source_and_proactivity(tmp_path):
    from bridge.rex_context_policy_bridge import handle_context_policy_request

    service, policy, _location, _knowledge, preferences = _service(tmp_path)
    policy.register_source(
        "integration:calendar",
        ContextSourceType.INTEGRATION,
        owner_user_id="james",
    )

    source_body, source_code = handle_context_policy_request(
        {
            "command": "set_source_context",
            "source_id": "integration:calendar",
            "enabled": False,
        },
        service=service,
        bound_user_id="james",
    )
    pref_body, pref_code = handle_context_policy_request(
        {"command": "set_proactive_assistance", "enabled": False},
        service=service,
        bound_user_id="james",
    )

    assert source_code == 0 and source_body["source"]["context_enabled"] is False
    assert pref_code == 0 and pref_body["proactive_assistance"] is False
    assert preferences.get("james").proactive_assistance is False


def test_state_lists_safe_owned_context_metadata(tmp_path):
    from bridge.rex_context_policy_bridge import handle_context_policy_request

    service, policy, location, knowledge, _preferences = _service(tmp_path)
    policy.register_source(
        "integration:calendar",
        ContextSourceType.INTEGRATION,
        owner_user_id="james",
    )
    knowledge.ingest_text(
        "secret contents must not appear",
        "Private Trip",
        owner_user_id="james",
        audience_scope="private",
        context_enabled=True,
    )
    location.set_assist(owner_user_id="james", enabled=True, actor_user_id="james")

    body, code = handle_context_policy_request(
        {"command": "get_state"},
        service=service,
        bound_user_id="james",
    )

    assert code == 0 and body["ok"] is True
    assert body["location"]["location_assist"] is True
    assert body["proactive_assistance"] is True
    assert body["uploads"][0]["title"] == "Private Trip"
    assert "secret contents" not in repr(body)
    assert any(item["source_id"] == "integration:calendar" for item in body["sources"])


def test_privacy_service_rejects_foreign_actor_even_without_transport(tmp_path):
    service, _policy, location, _knowledge, _preferences = _service(tmp_path)

    with pytest.raises(PermissionError, match="owner authorization"):
        service.set_location_assist(
            owner_user_id="cole",
            enabled=True,
            actor_user_id="james",
        )

    assert location.get("cole").location_assist is False


@pytest.mark.parametrize("caller_kind", ["openclaw", "self_maintenance", "developer_agent"])
def test_privileged_caller_labels_never_widen_privacy_authority(tmp_path, caller_kind):
    from bridge.rex_context_policy_bridge import handle_context_policy_request

    service, _policy, location, _knowledge, _preferences = _service(tmp_path)
    body, code = handle_context_policy_request(
        {
            "command": "set_location_assist",
            "user_id": "cole",
            "enabled": True,
            "caller_kind": caller_kind,
            "caller_permissions": ["admin", "self_modify"],
        },
        service=service,
        bound_user_id="james",
    )

    assert code != 0
    assert body["ok"] is False
    assert location.get("cole").location_assist is False
