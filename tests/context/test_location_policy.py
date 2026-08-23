from __future__ import annotations

from types import SimpleNamespace

import pytest

import rex.geolocation as geo
from rex.assistant import Assistant
from rex.context.location_policy import (
    LocationContextService,
    LocationGrantStore,
    LocationUsePurpose,
    PrivateLocation,
)
from rex.context.source_policy import ContextSourcePolicyStore


def _service(tmp_path, *, provider=None):
    source_policy = ContextSourcePolicyStore(tmp_path / "policy")
    grants = LocationGrantStore(
        tmp_path / "grants",
        source_policy_store=source_policy,
    )
    service = LocationContextService(grant_store=grants, location_provider=provider)
    return service, grants, source_policy


def test_admin_cannot_enable_another_users_location(tmp_path):
    _service_obj, grants, _source_policy = _service(tmp_path)

    with pytest.raises(PermissionError, match="owner authorization required"):
        grants.set_assist(owner_user_id="cole", enabled=True, actor_user_id="james")


def test_admin_cannot_share_another_users_location(tmp_path):
    _service_obj, grants, _source_policy = _service(tmp_path)

    with pytest.raises(PermissionError, match="owner authorization required"):
        grants.set_share(
            owner_user_id="cole",
            recipient_user_id="james",
            enabled=True,
            actor_user_id="james",
        )


def test_denied_disclosure_does_not_confirm_location_presence(tmp_path):
    service, grants, _source_policy = _service(tmp_path)
    grants.set_assist(owner_user_id="cole", enabled=True, actor_user_id="cole")
    service.seed_private_location(
        "cole",
        city="Dallas",
        timezone="America/Chicago",
        lat=32.7767,
        lon=-96.797,
    )

    result = service.get_for_disclosure(subject_user_id="cole", requester_user_id="james")

    assert result.allowed is False
    assert result.location is None
    assert result.message == "I can't share Cole's location."
    assert "Dallas" not in result.message


def test_assistance_requires_owner_opt_in_before_provider_is_called(tmp_path):
    calls: list[tuple[str, LocationUsePurpose]] = []

    def provider(user_id: str, purpose: LocationUsePurpose) -> PrivateLocation:
        calls.append((user_id, purpose))
        return PrivateLocation(city="Dallas", timezone="America/Chicago")

    service, grants, _source_policy = _service(tmp_path, provider=provider)

    assert service.get_for_assistance("james", LocationUsePurpose.TOOL_CONTEXT) is None
    assert calls == []

    grants.set_assist(owner_user_id="james", enabled=True, actor_user_id="james")
    location = service.get_for_assistance("james", LocationUsePurpose.TOOL_CONTEXT)

    assert location is not None
    assert location.city == "Dallas"
    assert calls == [("james", LocationUsePurpose.TOOL_CONTEXT)]


def test_recipient_share_is_separate_and_assist_disable_wins(tmp_path):
    service, grants, _source_policy = _service(tmp_path)
    grants.set_assist(owner_user_id="cole", enabled=True, actor_user_id="cole")
    service.seed_private_location("cole", city="Dallas", timezone="America/Chicago")

    denied = service.get_for_disclosure(subject_user_id="cole", requester_user_id="james")
    assert denied.allowed is False

    grants.set_share(
        owner_user_id="cole",
        recipient_user_id="james",
        enabled=True,
        actor_user_id="cole",
    )
    allowed = service.get_for_disclosure(subject_user_id="cole", requester_user_id="james")
    assert allowed.allowed is True
    assert allowed.location is not None
    assert allowed.location.city == "Dallas"

    grants.set_assist(owner_user_id="cole", enabled=False, actor_user_id="cole")
    disabled = service.get_for_disclosure(subject_user_id="cole", requester_user_id="james")
    assert disabled.allowed is False
    assert disabled.location is None
    assert disabled.message == "I can't share Cole's location."


def test_one_users_grant_and_location_do_not_affect_another(tmp_path):
    service, grants, _source_policy = _service(tmp_path)
    grants.set_assist(owner_user_id="james", enabled=True, actor_user_id="james")
    service.seed_private_location("james", city="Dallas", timezone="America/Chicago")
    service.seed_private_location("cole", city="Austin", timezone="America/Chicago")

    james = service.get_for_assistance("james", LocationUsePurpose.EXPLICIT_REQUEST)
    cole = service.get_for_assistance("cole", LocationUsePurpose.EXPLICIT_REQUEST)

    assert james is not None and james.city == "Dallas"
    assert cole is None


def test_location_grant_changes_bump_context_source_revision(tmp_path):
    _service_obj, grants, source_policy = _service(tmp_path)
    before = source_policy.revision_for_user("cole")

    grants.set_assist(owner_user_id="cole", enabled=True, actor_user_id="cole")
    after_assist = source_policy.revision_for_user("cole")
    assert after_assist != before

    grants.set_share(
        owner_user_id="cole",
        recipient_user_id="james",
        enabled=True,
        actor_user_id="cole",
    )
    after_share = source_policy.revision_for_user("cole")
    assert after_share != after_assist


def test_ambient_ip_cache_is_not_personal_tool_context(monkeypatch):
    monkeypatch.setattr(
        geo,
        "_location_cache",
        {"city": "Dallas", "timezone": "America/Chicago", "lat": 1.0, "lon": 2.0},
    )
    assistant = Assistant.__new__(Assistant)
    assistant._settings = SimpleNamespace(default_location=None, default_timezone=None)
    assistant._location_context_service = None

    assert assistant._build_tool_context() == {}


def test_user_bound_personal_location_overrides_static_tool_context():
    class StubLocationService:
        def get_for_assistance(self, user_id, purpose):
            assert user_id == "james"
            assert purpose is LocationUsePurpose.TOOL_CONTEXT
            return PrivateLocation(city="Austin", timezone="America/Chicago")

    assistant = Assistant.__new__(Assistant)
    assistant._settings = SimpleNamespace(
        default_location="Dallas, TX",
        default_timezone="America/Chicago",
    )
    assistant._location_context_service = StubLocationService()

    context = assistant._build_tool_context("james")

    assert context == {"location": "Austin", "timezone": "America/Chicago"}


def test_stale_location_is_refreshed_only_after_authorization(tmp_path):
    now = [100.0]
    provider_calls: list[str] = []

    def provider(user_id: str, purpose: LocationUsePurpose) -> PrivateLocation:
        provider_calls.append(user_id)
        return PrivateLocation(city="Austin", timezone="America/Chicago")

    source_policy = ContextSourcePolicyStore(tmp_path / "policy")
    grants = LocationGrantStore(
        tmp_path / "grants",
        source_policy_store=source_policy,
    )
    service = LocationContextService(
        grant_store=grants,
        location_provider=provider,
        clock=lambda: now[0],
        max_location_age_seconds=300.0,
    )
    service.seed_private_location("james", city="Dallas", timezone="America/Chicago")
    now[0] = 401.0

    assert service.get_for_assistance("james", LocationUsePurpose.TOOL_CONTEXT) is None
    assert provider_calls == []

    grants.set_assist(owner_user_id="james", enabled=True, actor_user_id="james")
    refreshed = service.get_for_assistance("james", LocationUsePurpose.TOOL_CONTEXT)

    assert refreshed is not None and refreshed.city == "Austin"
    assert provider_calls == ["james"]
