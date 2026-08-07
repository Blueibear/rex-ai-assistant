"""Behavior and safety tests for the authenticated profile bridge."""

from __future__ import annotations

import base64
import io
import json
import sys
from typing import Any

import pytest

from bridge import rex_profile_bridge
from rex.user_profile_service import UserProfileView


class FakeProfileService:
    def __init__(self) -> None:
        self.preferences: dict[str, Any] = {}
        self.avatar: tuple[bytes, str] | None = None
        self.removed = False

    def get_profile(self, user_id: str) -> UserProfileView:
        return UserProfileView(
            user_id=user_id,
            name="Test User",
            initials="TU",
            role="Administrator",
            permissions=["admin"],
            preferences=dict(self.preferences),
            voice_enrolled=True,
            voice_model_id="voice-model",
            voice_sample_count=3,
            voice_updated_at="2026-08-06T00:00:00Z",
            avatar_present=self.avatar is not None,
            avatar_mime_type=self.avatar[1] if self.avatar else None,
            avatar_data=base64.b64encode(self.avatar[0]).decode() if self.avatar else None,
            scope_labels={"profile": "user-private", "household_settings": "shared"},
        )

    def update_preferences(self, user_id: str, preferences: dict[str, Any]) -> None:
        assert user_id == "alice"
        self.preferences.update(preferences)

    def set_avatar(self, user_id: str, data: bytes, mime_type: str) -> None:
        assert user_id == "alice"
        self.avatar = (data, mime_type)

    def remove_avatar(self, user_id: str) -> None:
        assert user_id == "alice"
        self.avatar = None
        self.removed = True


@pytest.fixture
def service() -> FakeProfileService:
    return FakeProfileService()


def payload(action: str, **extra: object) -> dict[str, object]:
    return {
        "action": action,
        "user": "alice",
        "session_id": "session-1",
        "data_scope": "private",
        **extra,
    }


def test_get_serializes_complete_profile(service: FakeProfileService) -> None:
    response, code = rex_profile_bridge.process_payload(payload("get"), service=service)

    assert code == 0
    assert response["ok"] is True
    profile = response["profile"]
    assert profile["user_id"] == "alice"
    assert profile["permissions"] == ["admin"]
    assert profile["voice_sample_count"] == 3
    assert profile["initials"] == "TU"
    assert profile["scope_labels"]["household_settings"] == "shared"


def test_mutations_return_refreshed_profile(service: FakeProfileService) -> None:
    response, code = rex_profile_bridge.process_payload(
        payload("update_preferences", preferences={"theme": "dark"}), service=service
    )
    assert code == 0
    assert response["profile"]["preferences"] == {"theme": "dark"}

    encoded = base64.b64encode(b"avatar-bytes").decode("ascii")
    response, code = rex_profile_bridge.process_payload(
        payload("set_avatar", mime_type="image/png", avatar_base64=encoded), service=service
    )
    assert code == 0
    assert response["profile"]["avatar_present"] is True
    assert service.avatar == (b"avatar-bytes", "image/png")

    response, code = rex_profile_bridge.process_payload(payload("remove_avatar"), service=service)
    assert code == 0
    assert service.removed is True
    assert response["profile"]["avatar_present"] is False


@pytest.mark.parametrize("scope", [None, "shared_household", "public"])
def test_private_scope_is_required(service: FakeProfileService, scope: object) -> None:
    request_payload = payload("get")
    request_payload["data_scope"] = scope
    response, code = rex_profile_bridge.process_payload(request_payload, service=service)

    assert code == 1
    assert response == {"ok": False, "error": "Permission denied"}


@pytest.mark.parametrize("user", [None, 123, "", "..", "a/b"])
def test_user_must_be_a_valid_string(service: FakeProfileService, user: object) -> None:
    request_payload = payload("get")
    request_payload["user"] = user
    response, code = rex_profile_bridge.process_payload(request_payload, service=service)

    assert code == 1
    assert response == {"ok": False, "error": "Request validation failed"}


@pytest.mark.parametrize("field", ["user_id", "target_user", "target_user_id"])
def test_cross_user_authority_fields_are_rejected(service: FakeProfileService, field: str) -> None:
    response, code = rex_profile_bridge.process_payload(
        payload("update_preferences", preferences={}, **{field: "bob"}), service=service
    )

    assert code == 1
    assert response == {"ok": False, "error": "Permission denied"}


def test_matching_authority_field_does_not_change_session_user(
    service: FakeProfileService,
) -> None:
    response, code = rex_profile_bridge.process_payload(
        payload("update_preferences", preferences={"theme": "dark"}, user_id="alice"),
        service=service,
    )
    assert code == 0
    assert response["profile"]["user_id"] == "alice"


@pytest.mark.parametrize(
    "request_payload",
    [
        None,
        [],
        "text",
        {},
        payload("unsupported"),
        payload("update_preferences", preferences=[]),
    ],
)
def test_malformed_requests_fail_with_fixed_errors(
    service: FakeProfileService, request_payload: object
) -> None:
    response, code = rex_profile_bridge.process_payload(request_payload, service=service)

    assert code == 1
    assert response == {"ok": False, "error": "Request validation failed"}
    assert "traceback" not in response


@pytest.mark.parametrize(
    "encoded",
    ["", "not-base64!", "abc", pytest.param("a" * 2_900_001, id="oversized")],
)
def test_avatar_base64_is_strictly_bounded(service: FakeProfileService, encoded: str) -> None:
    response, code = rex_profile_bridge.process_payload(
        payload("set_avatar", mime_type="image/jpeg", avatar_base64=encoded),
        service=service,
    )

    assert code == 1
    assert response == {"ok": False, "error": "Request validation failed"}


@pytest.mark.parametrize("mime", [None, "", "image/gif", "text/plain"])
def test_avatar_mime_type_is_restricted(service: FakeProfileService, mime: object) -> None:
    response, code = rex_profile_bridge.process_payload(
        payload(
            "set_avatar",
            mime_type=mime,
            avatar_base64=base64.b64encode(b"data").decode("ascii"),
        ),
        service=service,
    )

    assert code == 1
    assert response == {"ok": False, "error": "Request validation failed"}


class ExplodingService(FakeProfileService):
    def get_profile(self, user_id: str) -> UserProfileView:
        raise OSError(r"C:\private\profile.jpg token=secret-marker")


def test_service_errors_never_leak_private_details() -> None:
    response, code = rex_profile_bridge.process_payload(payload("get"), service=ExplodingService())

    assert code == 1
    assert response == {"ok": False, "error": "Profile operation failed"}
    serialized = json.dumps(response)
    assert "private" not in serialized
    assert "secret-marker" not in serialized
    assert "traceback" not in serialized


def test_main_rejects_invalid_json_without_echoing_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"secret":"marker"'))  # pragma: allowlist secret

    with pytest.raises(SystemExit) as exc_info:
        rex_profile_bridge.main()

    assert exc_info.value.code == 1
    response = json.loads(capsys.readouterr().out)
    assert response == {"ok": False, "error": "Request validation failed"}
    assert "marker" not in json.dumps(response)
