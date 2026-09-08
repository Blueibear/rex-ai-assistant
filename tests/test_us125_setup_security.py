"""Security regressions for the US-125 first-run setup boundary."""

from __future__ import annotations

import json
from typing import Any

import pytest

from bridge import rex_setup_bridge


class _ExistingUsersCursor:
    def fetchone(self) -> tuple[int]:
        return (1,)


class _ExistingUsersDb:
    def __enter__(self) -> "_ExistingUsersDb":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def execute(self, query: str, _params: object = ()) -> _ExistingUsersCursor:
        assert "SELECT COUNT(*) FROM users" in query
        return _ExistingUsersCursor()


def test_complete_setup_rejects_repeat_first_run_mutation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    create_calls: list[tuple[str, str]] = []
    bootstrap_calls: list[str] = []
    persist_calls: list[dict[str, str]] = []

    monkeypatch.setattr("rex.auth._open_db", lambda: _ExistingUsersDb())

    def fake_create_user(username: str, password: str) -> dict[str, str]:
        create_calls.append((username, password))
        return {"id": "unexpected-user"}

    def fake_persist(
        values: dict[str, str], *, config_path: object = None, update_config: Any = None
    ) -> dict[str, str]:
        del config_path, update_config
        persist_calls.append(dict(values))
        return {}

    monkeypatch.setattr("rex.auth.create_user", fake_create_user)
    monkeypatch.setattr("rex.permissions.bootstrap_admin_if_first_user", bootstrap_calls.append)
    monkeypatch.setattr("rex.credential_persistence.persist_household_secrets", fake_persist)

    rex_setup_bridge._handle_complete(
        {
            "username": "james",
            "password": "securepass1",  # pragma: allowlist secret
            "llm_provider": "local",
            "tts_provider": "pyttsx3",
            "wake_word_id": "hey_rex",
            "room_name": "Office",
            "background_voice_enabled": False,
            "ha_base_url": "",
            "ha_token": "",
        }
    )

    response = json.loads(capsys.readouterr().out)
    assert response == {"ok": False, "error": "setup is already complete"}
    assert create_calls == []
    assert bootstrap_calls == []
    assert persist_calls == []
