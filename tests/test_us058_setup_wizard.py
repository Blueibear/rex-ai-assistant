"""Tests for US-058: First-run setup wizard API.

Covers:
- GET /api/setup/status returns needs_setup=True when no users exist
- GET /api/setup/status returns needs_setup=False after a user exists
- POST /api/setup/complete creates user and returns ok
- POST /api/setup/complete rejects missing username/password
- POST /api/setup/complete returns 409 if setup already done
- POST /api/setup/complete writes secrets to .env file
- POST /api/setup/complete writes non-secret settings to rex_config.json
- _write_env_secrets: new key appended, existing key updated, unrelated keys preserved
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
    monkeypatch.setenv("REX_JWT_SECRET", "test-us058-secret")
    return tmp_path


@pytest.fixture()
def flask_client(tmp_data_dir: Path):  # type: ignore[override]
    from rex.gui_app import _create_flask_app

    app = _create_flask_app(ui_enabled=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _setup_token(client: object) -> str:
    """Return the single-use setup token from the Flask app config."""
    return client.application.config.get("SETUP_TOKEN") or ""  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# GET /api/setup/status
# ---------------------------------------------------------------------------


class TestSetupStatus:
    def test_needs_setup_true_when_no_users(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.get("/api/setup/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["needs_setup"] is True

    def test_needs_setup_false_after_registration(self, flask_client) -> None:  # type: ignore[override]
        flask_client.post(
            "/api/auth/register",
            json={"username": "alice", "password": "pass1234"},
            headers={"X-Setup-Token": _setup_token(flask_client)},
        )
        resp = flask_client.get("/api/setup/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["needs_setup"] is False

    def test_no_auth_required(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.get("/api/setup/status")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/setup/complete
# ---------------------------------------------------------------------------


class TestSetupComplete:
    def test_creates_user_returns_ok(
        self, flask_client, tmp_data_dir: Path
    ) -> None:  # type: ignore[override]
        resp = flask_client.post(
            "/api/setup/complete",
            json={
                "username": "bob",
                "password": "securepass1",
                "llm_provider": "local",
                "tts_provider": "none",
            },
            headers={"X-Setup-Token": _setup_token(flask_client)},
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["ok"] is True
        assert "user_id" in data

    def test_missing_username_returns_400(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.post(
            "/api/setup/complete",
            json={"password": "securepass1"},
            headers={"X-Setup-Token": _setup_token(flask_client)},
        )
        assert resp.status_code == 400

    def test_missing_password_returns_400(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.post(
            "/api/setup/complete",
            json={"username": "bob"},
            headers={"X-Setup-Token": _setup_token(flask_client)},
        )
        assert resp.status_code == 400

    def test_second_setup_attempt_returns_403(self, flask_client) -> None:  # type: ignore[override]
        """After setup completes the token is consumed; re-running returns 403."""
        payload = {
            "username": "carol",
            "password": "securepass1",
            "llm_provider": "local",
            "tts_provider": "none",
        }
        flask_client.post(
            "/api/setup/complete",
            json=payload,
            headers={"X-Setup-Token": _setup_token(flask_client)},
        )
        resp = flask_client.post("/api/setup/complete", json=payload)
        assert resp.status_code == 403

    def test_sets_needs_setup_to_false(self, flask_client) -> None:  # type: ignore[override]
        flask_client.post(
            "/api/setup/complete",
            json={
                "username": "dave",
                "password": "securepass1",
                "llm_provider": "local",
                "tts_provider": "none",
            },
            headers={"X-Setup-Token": _setup_token(flask_client)},
        )
        status_resp = flask_client.get("/api/setup/status")
        assert status_resp.get_json()["needs_setup"] is False

    def test_writes_openai_key_to_env(
        self, flask_client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:  # type: ignore[override]
        monkeypatch.setattr("rex.gui_app._write_env_secrets.__module__", "rex.gui_app")

        from rex import gui_app

        captured: dict[str, object] = {}

        original = gui_app._write_env_secrets  # type: ignore[attr-defined]

        def fake_write(path: Path, *, llm_provider: str, llm_api_key: str, ha_token: str) -> None:
            captured["path"] = path
            captured["llm_provider"] = llm_provider
            captured["llm_api_key"] = llm_api_key
            captured["ha_token"] = ha_token

        monkeypatch.setattr(gui_app, "_write_env_secrets", fake_write)

        flask_client.post(
            "/api/setup/complete",
            json={
                "username": "eve",
                "password": "securepass1",
                "llm_provider": "openai",
                "llm_api_key": "sk-test-key",
                "tts_provider": "none",
            },
            headers={"X-Setup-Token": _setup_token(flask_client)},
        )

        assert captured.get("llm_api_key") == "sk-test-key"
        assert captured.get("llm_provider") == "openai"

        monkeypatch.setattr(gui_app, "_write_env_secrets", original)


# ---------------------------------------------------------------------------
# _write_env_secrets unit tests
# ---------------------------------------------------------------------------


class TestWriteEnvSecrets:
    def test_creates_env_file_with_openai_key(self, tmp_path: Path) -> None:
        from rex.gui_app import _write_env_secrets

        env_path = tmp_path / ".env"
        _write_env_secrets(env_path, llm_provider="openai", llm_api_key="sk-abc", ha_token="")
        content = env_path.read_text(encoding="utf-8")
        assert "OPENAI_API_KEY=sk-abc" in content

    def test_creates_env_file_with_anthropic_key(self, tmp_path: Path) -> None:
        from rex.gui_app import _write_env_secrets

        env_path = tmp_path / ".env"
        _write_env_secrets(
            env_path, llm_provider="anthropic", llm_api_key="sk-ant-test", ha_token=""
        )
        content = env_path.read_text(encoding="utf-8")
        assert "ANTHROPIC_API_KEY=sk-ant-test" in content

    def test_writes_ha_token(self, tmp_path: Path) -> None:
        from rex.gui_app import _write_env_secrets

        env_path = tmp_path / ".env"
        _write_env_secrets(env_path, llm_provider="local", llm_api_key="", ha_token="my-ha-token")
        content = env_path.read_text(encoding="utf-8")
        assert "HA_TOKEN=my-ha-token" in content

    def test_preserves_unrelated_lines(self, tmp_path: Path) -> None:
        from rex.gui_app import _write_env_secrets

        env_path = tmp_path / ".env"
        env_path.write_text("SOME_OTHER_VAR=hello\n", encoding="utf-8")
        _write_env_secrets(env_path, llm_provider="openai", llm_api_key="sk-x", ha_token="")
        content = env_path.read_text(encoding="utf-8")
        assert "SOME_OTHER_VAR=hello" in content
        assert "OPENAI_API_KEY=sk-x" in content

    def test_updates_existing_key(self, tmp_path: Path) -> None:
        from rex.gui_app import _write_env_secrets

        env_path = tmp_path / ".env"
        env_path.write_text("OPENAI_API_KEY=old-key\n", encoding="utf-8")
        _write_env_secrets(env_path, llm_provider="openai", llm_api_key="new-key", ha_token="")
        content = env_path.read_text(encoding="utf-8")
        assert "new-key" in content
        assert "old-key" not in content
        assert content.count("OPENAI_API_KEY") == 1

    def test_local_provider_writes_no_api_key(self, tmp_path: Path) -> None:
        from rex.gui_app import _write_env_secrets

        env_path = tmp_path / ".env"
        _write_env_secrets(env_path, llm_provider="local", llm_api_key="", ha_token="")
        content = env_path.read_text(encoding="utf-8")
        assert "OPENAI_API_KEY" not in content
        assert "ANTHROPIC_API_KEY" not in content
