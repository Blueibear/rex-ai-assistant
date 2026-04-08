"""Tests for US-061: Command history.

Covers:
- CommandHistoryStore: record / get_recent / clear
- get_recent respects limit and returns newest first
- get_recent clamps limit to 1–500
- GET /api/history requires auth
- GET /api/history returns history list with expected keys
- GET /api/history?limit=5 returns at most 5 entries
- success field is bool
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Unit: CommandHistoryStore
# ---------------------------------------------------------------------------


class TestCommandHistoryStore:
    def test_record_returns_positive_id(self, tmp_path: Path) -> None:
        from rex.command_history import CommandHistoryStore

        store = CommandHistoryStore(db_path=tmp_path / "h.db")
        row_id = store.record("hello rex")
        assert row_id > 0

    def test_get_recent_returns_recorded_entries(self, tmp_path: Path) -> None:
        from rex.command_history import CommandHistoryStore

        store = CommandHistoryStore(db_path=tmp_path / "h.db")
        store.record("cmd one", result="res one", success=True)
        store.record("cmd two", result="res two", success=False)

        entries = store.get_recent()
        assert len(entries) == 2
        # newest first
        assert entries[0]["command"] == "cmd two"
        assert entries[1]["command"] == "cmd one"

    def test_success_field_is_bool(self, tmp_path: Path) -> None:
        from rex.command_history import CommandHistoryStore

        store = CommandHistoryStore(db_path=tmp_path / "h.db")
        store.record("ok cmd", success=True)
        store.record("fail cmd", success=False)
        entries = store.get_recent()
        for e in entries:
            assert isinstance(e["success"], bool)

    def test_get_recent_respects_limit(self, tmp_path: Path) -> None:
        from rex.command_history import CommandHistoryStore

        store = CommandHistoryStore(db_path=tmp_path / "h.db")
        for i in range(10):
            store.record(f"cmd {i}")
        entries = store.get_recent(limit=3)
        assert len(entries) == 3

    def test_get_recent_clamps_limit_min(self, tmp_path: Path) -> None:
        from rex.command_history import CommandHistoryStore

        store = CommandHistoryStore(db_path=tmp_path / "h.db")
        store.record("only one")
        entries = store.get_recent(limit=0)
        assert len(entries) >= 1  # clamped to 1

    def test_get_recent_clamps_limit_max(self, tmp_path: Path) -> None:
        from rex.command_history import CommandHistoryStore

        store = CommandHistoryStore(db_path=tmp_path / "h.db")
        # 600 entries
        for i in range(600):
            store.record(f"cmd {i}")
        entries = store.get_recent(limit=600)
        assert len(entries) == 500  # clamped to 500

    def test_clear_removes_all_entries(self, tmp_path: Path) -> None:
        from rex.command_history import CommandHistoryStore

        store = CommandHistoryStore(db_path=tmp_path / "h.db")
        store.record("cmd a")
        store.record("cmd b")
        store.clear()
        assert store.get_recent() == []

    def test_entry_has_required_keys(self, tmp_path: Path) -> None:
        from rex.command_history import CommandHistoryStore

        store = CommandHistoryStore(db_path=tmp_path / "h.db")
        store.record("test cmd", result="test result", success=True)
        entry = store.get_recent()[0]
        assert "id" in entry
        assert "timestamp" in entry
        assert "command" in entry
        assert "result" in entry
        assert "success" in entry


# ---------------------------------------------------------------------------
# API: GET /api/history
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("REX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REX_JWT_SECRET", "test-us061-secret-long-enough-32chars")
    return tmp_path


@pytest.fixture()
def flask_client(tmp_data_dir: Path):  # type: ignore[override]
    from rex.gui_app import _create_flask_app

    app = _create_flask_app(ui_enabled=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _register_and_login(client, username: str = "admin", password: str = "pass1234x!") -> str:
    client.post("/api/auth/register", json={"username": username, "password": password})
    resp = client.post("/api/auth/login", json={"username": username, "password": password})
    return resp.get_json()["token"]  # type: ignore[index]


class TestHistoryEndpoint:
    def test_requires_auth(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.get("/api/history")
        assert resp.status_code == 401

    def test_returns_history_key(self, flask_client) -> None:  # type: ignore[override]
        token = _register_and_login(flask_client)
        resp = flask_client.get("/api/history", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "history" in data
        assert isinstance(data["history"], list)

    def test_limit_param_respected(
        self, flask_client, tmp_data_dir: Path
    ) -> None:  # type: ignore[override]
        token = _register_and_login(flask_client)

        from rex.command_history import CommandHistoryStore

        store = CommandHistoryStore(db_path=tmp_data_dir / "command_history.db")
        for i in range(20):
            store.record(f"cmd {i}")

        resp = flask_client.get(
            "/api/history?limit=5", headers={"Authorization": f"Bearer {token}"}
        )
        data = resp.get_json()
        assert len(data["history"]) == 5

    def test_entries_have_required_fields(
        self, flask_client, tmp_data_dir: Path
    ) -> None:  # type: ignore[override]
        token = _register_and_login(flask_client)

        from rex.command_history import CommandHistoryStore

        store = CommandHistoryStore(db_path=tmp_data_dir / "command_history.db")
        store.record("voice command", result="done", success=True)

        resp = flask_client.get("/api/history", headers={"Authorization": f"Bearer {token}"})
        data = resp.get_json()
        assert len(data["history"]) >= 1
        entry = data["history"][0]
        for key in ("id", "timestamp", "command", "result", "success"):
            assert key in entry
