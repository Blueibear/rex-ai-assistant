"""Current-session endpoint and live authorization projection tests.

Matrix rows: SES-001, SES-005..SES-007, AUTH-021..AUTH-023.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.mobile_api.conftest import auth_header, create_user, login_tokens

_SESSION_URL = "/mobile/auth/session"


class TestCurrentSession:
    def test_returns_live_projection(self, client) -> None:
        user_id = create_user("james", "pw-123456", admin=True)
        tokens = login_tokens(client, "james", "pw-123456")
        response = client.get(_SESSION_URL, headers=auth_header(tokens["access_token"]))
        assert response.status_code == 200
        body = response.get_json()
        assert body["session_id"] == tokens["session_id"]
        assert body["user"]["id"] == user_id
        assert body["user"]["role"] == "owner"
        assert body["user"]["permissions"] == ["admin"]

    def test_client_user_id_is_ignored(self, client) -> None:
        """SES-005: a user_id query parameter never changes the principal."""
        user_id = create_user("james", "pw-123456")
        other_id = create_user("sarah", "pw-abcdef")
        tokens = login_tokens(client, "james", "pw-123456")
        response = client.get(
            f"{_SESSION_URL}?user_id={other_id}",
            headers=auth_header(tokens["access_token"]),
        )
        assert response.status_code == 200
        assert response.get_json()["user"]["id"] == user_id

    def test_non_admin_maps_to_member(self, client) -> None:
        """SES-006: role is a presentation projection of live permissions."""
        create_user("james", "pw-123456", admin=False)
        tokens = login_tokens(client, "james", "pw-123456")
        response = client.get(_SESSION_URL, headers=auth_header(tokens["access_token"]))
        assert response.get_json()["user"]["role"] == "member"

    def test_permission_change_reflected_live(self, client) -> None:
        """SES-007 / AUTH-022: current DB permissions, not token-time claims."""
        from rex.permissions import grant_permission, revoke_permission

        user_id = create_user("james", "pw-123456", admin=True)
        tokens = login_tokens(client, "james", "pw-123456")
        access = tokens["access_token"]

        grant_permission(user_id, "ha_control")
        first = client.get(_SESSION_URL, headers=auth_header(access)).get_json()
        assert "ha_control" in first["user"]["permissions"]

        revoke_permission(user_id, "admin")
        second = client.get(_SESSION_URL, headers=auth_header(access)).get_json()
        assert second["user"]["role"] == "member"
        assert "admin" not in second["user"]["permissions"]

    def test_profile_change_reflected_live(self, client, monkeypatch, tmp_path: Path) -> None:
        """AUTH-023: /session reflects the current display profile."""
        from rex import identity as rex_identity
        from rex.mobile_api import auth as mauth
        from rex.mobile_api import users as musers

        user_id = create_user("james", "pw-123456")
        profile_dir = tmp_path / "Memory" / user_id
        profile_dir.mkdir(parents=True)

        def _fake_profile(requested_id: str, *, memory_dir=None):
            rex_identity.validate_user_id(requested_id)
            core = tmp_path / "Memory" / requested_id / "core.json"
            if core.exists():
                return json.loads(core.read_text(encoding="utf-8"))
            return None

        monkeypatch.setattr(musers, "get_user_profile", _fake_profile)
        monkeypatch.setattr(mauth, "get_user_profile", _fake_profile)

        tokens = login_tokens(client, "james", "pw-123456")
        access = tokens["access_token"]

        (profile_dir / "core.json").write_text(json.dumps({"name": "James R."}), encoding="utf-8")
        response = client.get(_SESSION_URL, headers=auth_header(access))
        assert response.get_json()["user"]["name"] == "James R."
