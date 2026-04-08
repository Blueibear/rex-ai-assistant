"""Tests for US-063: Quick actions panel.

Covers:
- GET /api/quick-actions returns list (auth required)
- GET /api/quick-actions returns 401 without token
- POST /api/quick-actions creates action with id, label, command
- POST /api/quick-actions returns 400 when label or command missing
- DELETE /api/quick-actions/<id> removes action
- DELETE /api/quick-actions/<id> returns 404 for unknown id
- POST /api/quick-actions/<id>/run returns reply
- POST /api/quick-actions/<id>/run returns 404 for unknown id
- Multiple actions can be created and listed
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("REX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REX_JWT_SECRET", "test-us063-secret-long-enough-32chars")
    return tmp_path


@pytest.fixture()
def flask_client(tmp_data_dir: Path):  # type: ignore[override]
    from rex.gui_app import _create_flask_app

    app = _create_flask_app(ui_enabled=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _register_and_login(client, username: str = "testuser", password: str = "TestPass123!") -> str:
    client.post("/api/auth/register", json={"username": username, "password": password})
    resp = client.post("/api/auth/login", json={"username": username, "password": password})
    return resp.get_json()["token"]  # type: ignore[index]


@pytest.fixture()
def auth_token(flask_client) -> str:  # type: ignore[override]
    """Register a user and return a JWT token."""
    return _register_and_login(flask_client)


def auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------


class TestQuickActionsAuth:
    def test_get_requires_auth(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.get("/api/quick-actions")
        assert resp.status_code == 401

    def test_post_requires_auth(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.post(
            "/api/quick-actions",
            json={"label": "Test", "command": "do something"},
        )
        assert resp.status_code == 401

    def test_delete_requires_auth(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.delete("/api/quick-actions/some-id")
        assert resp.status_code == 401

    def test_run_requires_auth(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.post("/api/quick-actions/some-id/run")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestQuickActionsCRUD:
    def test_get_returns_empty_list_initially(self, flask_client, auth_token: str) -> None:
        resp = flask_client.get("/api/quick-actions", headers=auth_header(auth_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert "quick_actions" in data
        assert isinstance(data["quick_actions"], list)
        assert len(data["quick_actions"]) == 0

    def test_post_creates_action(self, flask_client, auth_token: str) -> None:
        resp = flask_client.post(
            "/api/quick-actions",
            json={"label": "Lights off", "command": "Turn off all lights"},
            headers=auth_header(auth_token),
        )
        assert resp.status_code in (200, 201)
        action = resp.get_json()
        assert action["label"] == "Lights off"
        assert action["command"] == "Turn off all lights"
        assert "id" in action
        assert len(action["id"]) > 0

    def test_post_returns_400_when_label_missing(self, flask_client, auth_token: str) -> None:
        resp = flask_client.post(
            "/api/quick-actions",
            json={"command": "do something"},
            headers=auth_header(auth_token),
        )
        assert resp.status_code == 400

    def test_post_returns_400_when_command_missing(self, flask_client, auth_token: str) -> None:
        resp = flask_client.post(
            "/api/quick-actions",
            json={"label": "My action"},
            headers=auth_header(auth_token),
        )
        assert resp.status_code == 400

    def test_created_action_appears_in_list(self, flask_client, auth_token: str) -> None:
        flask_client.post(
            "/api/quick-actions",
            json={"label": "Say hello", "command": "Say hello"},
            headers=auth_header(auth_token),
        )
        resp = flask_client.get("/api/quick-actions", headers=auth_header(auth_token))
        data = resp.get_json()
        labels = [a["label"] for a in data["quick_actions"]]
        assert "Say hello" in labels

    def test_multiple_actions_can_be_created(self, flask_client, auth_token: str) -> None:
        flask_client.post(
            "/api/quick-actions",
            json={"label": "Action 1", "command": "cmd 1"},
            headers=auth_header(auth_token),
        )
        flask_client.post(
            "/api/quick-actions",
            json={"label": "Action 2", "command": "cmd 2"},
            headers=auth_header(auth_token),
        )
        resp = flask_client.get("/api/quick-actions", headers=auth_header(auth_token))
        data = resp.get_json()
        assert len(data["quick_actions"]) == 2

    def test_delete_removes_action(self, flask_client, auth_token: str) -> None:
        create_resp = flask_client.post(
            "/api/quick-actions",
            json={"label": "To delete", "command": "delete me"},
            headers=auth_header(auth_token),
        )
        action_id = create_resp.get_json()["id"]

        del_resp = flask_client.delete(
            f"/api/quick-actions/{action_id}",
            headers=auth_header(auth_token),
        )
        assert del_resp.status_code == 200

        list_resp = flask_client.get("/api/quick-actions", headers=auth_header(auth_token))
        ids = [a["id"] for a in list_resp.get_json()["quick_actions"]]
        assert action_id not in ids

    def test_delete_unknown_id_returns_404(self, flask_client, auth_token: str) -> None:
        resp = flask_client.delete(
            "/api/quick-actions/nonexistent-id",
            headers=auth_header(auth_token),
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Run action
# ---------------------------------------------------------------------------


class TestQuickActionsRun:
    def test_run_unknown_id_returns_404(self, flask_client, auth_token: str) -> None:
        resp = flask_client.post(
            "/api/quick-actions/nonexistent-id/run",
            headers=auth_header(auth_token),
        )
        assert resp.status_code == 404

    def test_run_returns_reply_key(
        self, flask_client, auth_token: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Create an action first.
        create_resp = flask_client.post(
            "/api/quick-actions",
            json={"label": "Greet", "command": "Say hello"},
            headers=auth_header(auth_token),
        )
        action_id = create_resp.get_json()["id"]

        # Stub the LLM so the test doesn't need a real model.
        monkeypatch.setattr(
            "rex.gui_app._generate_reply",
            lambda *args, **kwargs: "Hello there!",
        )

        run_resp = flask_client.post(
            f"/api/quick-actions/{action_id}/run",
            headers=auth_header(auth_token),
        )
        assert run_resp.status_code == 200
        data = run_resp.get_json()
        assert "reply" in data
        assert data["reply"] == "Hello there!"
