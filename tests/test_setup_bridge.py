"""Behavior tests for the canonical Electron setup bridge."""

from __future__ import annotations

import json
from typing import Any

import pytest

from bridge import rex_setup_bridge


def _install_setup_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], list[str]]:
    captured: dict[str, Any] = {}
    bootstrapped: list[str] = []

    def fake_create_user(username: str, password: str) -> dict[str, str]:
        assert username == "james"
        assert password == "securepass1"
        return {"id": "james"}

    def fake_persist(
        values: dict[str, str], *, config_path=None, update_config=None
    ) -> dict[str, str]:
        captured["values"] = dict(values)
        config: dict[str, Any] = {}
        if update_config is not None:
            update_config(config)
        captured["config"] = config
        return {}

    monkeypatch.setattr("rex.auth.create_user", fake_create_user)
    monkeypatch.setattr("rex.permissions.bootstrap_admin_if_first_user", bootstrapped.append)
    monkeypatch.setattr("rex.credential_persistence.persist_household_secrets", fake_persist)
    return captured, bootstrapped


def test_deferred_home_assistant_is_omitted_from_persistence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured, bootstrapped = _install_setup_fakes(monkeypatch)

    rex_setup_bridge._handle_complete(
        {
            "username": "james",
            "password": "securepass1",
            "llm_provider": "local",
            "tts_provider": "none",
            "ha_base_url": "http://ha.local:8123",
            "ha_token": "must-not-persist",
            "defer_home_assistant": True,
        }
    )

    assert json.loads(capsys.readouterr().out) == {"ok": True, "user_id": "james"}
    assert captured["values"] == {}
    assert "home_assistant" not in captured["config"]
    assert bootstrapped == ["james"]


def test_string_false_does_not_defer_home_assistant(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured, _ = _install_setup_fakes(monkeypatch)

    rex_setup_bridge._handle_complete(
        {
            "username": "james",
            "password": "securepass1",
            "llm_provider": "local",
            "tts_provider": "none",
            "ha_base_url": "http://ha.local:8123",
            "ha_token": "test-token",
            "defer_home_assistant": "false",
        }
    )

    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert captured["values"] == {"HA_TOKEN": "test-token"}
    assert captured["config"]["home_assistant"]["base_url"] == "http://ha.local:8123"
