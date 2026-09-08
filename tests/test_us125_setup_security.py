"""Security regressions for the US-125 first-run setup boundary."""

from __future__ import annotations

import json
import sys
from types import ModuleType
from typing import Any

import pytest

from bridge import rex_setup_bridge


class _ExistingUsersCursor:
    def fetchone(self) -> tuple[int]:
        return (1,)


class _ExistingUsersDb:
    def __enter__(self) -> _ExistingUsersDb:
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

    auth_module = ModuleType("rex.auth")
    auth_module._open_db = lambda: _ExistingUsersDb()  # type: ignore[attr-defined]

    def fake_create_user(username: str, password: str) -> dict[str, str]:
        create_calls.append((username, password))
        return {"id": "unexpected-user"}

    auth_module.create_user = fake_create_user  # type: ignore[attr-defined]

    permissions_module = ModuleType("rex.permissions")
    permissions_module.bootstrap_admin_if_first_user = bootstrap_calls.append  # type: ignore[attr-defined]

    def fake_persist(
        values: dict[str, str], *, config_path: object = None, update_config: Any = None
    ) -> dict[str, str]:
        del config_path, update_config
        persist_calls.append(dict(values))
        return {}

    credential_module = ModuleType("rex.credential_persistence")
    credential_module.persist_household_secrets = fake_persist  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "rex.auth", auth_module)
    monkeypatch.setitem(sys.modules, "rex.permissions", permissions_module)
    monkeypatch.setitem(sys.modules, "rex.credential_persistence", credential_module)

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
