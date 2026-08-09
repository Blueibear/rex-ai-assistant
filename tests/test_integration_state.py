"""Evidence-based integration-state contract tests."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from rex.commands.core import cmd_integrations
from rex.doctor import Status, check_integration_states
from rex.integration_state import (
    IntegrationState,
    build_integration_inventory,
    configured_state,
)

EXPECTED_STATES = {
    "unavailable",
    "unconfigured",
    "configured",
    "reachable",
    "authenticated",
    "degraded",
    "read_only",
    "write_capable",
    "write_tested",
    "verified",
}


def _config(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "ha_base_url": None,
        "ha_token": None,
        "email_provider": "none",
        "email_accounts": [],
        "calendar_provider": "none",
        "search_providers": "",
        "openai_api_key": None,
        "ollama_base_url": "",
        "telegram_bot_token": None,
        "telegram_chat_id": None,
        "push_provider": None,
        "push_token": None,
        "integrations": SimpleNamespace(openclaw_gateway_url=""),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_state_vocabulary_is_complete_and_stable() -> None:
    assert {state.value for state in IntegrationState} == EXPECTED_STATES


def test_configuration_alone_is_not_capability_evidence() -> None:
    assert configured_state(True) is IntegrationState.CONFIGURED
    item = next(
        entry
        for entry in build_integration_inventory(
            _config(ha_base_url="http://ha.local:8123", ha_token="token"), {}
        )
        if entry.key == "home_assistant"
    )
    assert item.configured is True
    assert item.state is IntegrationState.CONFIGURED
    assert item.read_capable is False
    assert item.write_capable is False


def test_outlook_is_unavailable_not_connected() -> None:
    items = build_integration_inventory(_config(email_provider="outlook"), {})
    email = next(item for item in items if item.key == "email")
    assert email.state is IntegrationState.UNAVAILABLE
    assert email.available is False
    assert "unavailable" in email.detail


def test_twilio_requires_complete_credentials(monkeypatch) -> None:
    monkeypatch.setenv("REX_ALLOW_PLAINTEXT_CREDENTIAL_FALLBACK", "1")
    partial = build_integration_inventory(
        _config(),
        {"TWILIO_ACCOUNT_SID": "sid", "TWILIO_AUTH_TOKEN": "token"},
    )
    assert (
        next(item for item in partial if item.key == "sms").state is IntegrationState.UNCONFIGURED
    )

    complete = build_integration_inventory(
        _config(),
        {
            "TWILIO_ACCOUNT_SID": "sid",
            "TWILIO_AUTH_TOKEN": "token",
            "TWILIO_FROM_NUMBER": "+15550000000",
        },
    )
    assert next(item for item in complete if item.key == "sms").state is IntegrationState.CONFIGURED


def test_inventory_preserves_all_visible_integrations() -> None:
    keys = {item.key for item in build_integration_inventory(_config(), {})}
    assert {
        "home_assistant",
        "email",
        "calendar",
        "sms",
        "phone",
        "telegram",
        "search",
        "mqtt",
        "openai",
        "ollama",
        "push",
        "openclaw",
    } <= keys


def test_cli_inventory_never_labels_credentials_connected(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "rex.config.load_config",
        lambda: _config(ha_base_url="http://ha.local:8123", ha_token="token"),
    )
    assert cmd_integrations(Namespace()) == 0
    output = capsys.readouterr().out.lower()
    assert "home assistant: configured" in output
    assert "connected" not in output


def test_doctor_reports_configuration_as_info(monkeypatch) -> None:
    monkeypatch.setattr(
        "rex.config.load_config",
        lambda: _config(ha_base_url="http://ha.local:8123", ha_token="token"),
    )
    result = check_integration_states()
    assert result.status is Status.INFO
    assert "live state requires explicit provider tests" in result.message
    assert "Home Assistant: configured" in result.details


def test_gui_uses_the_same_state_vocabulary_and_has_no_email_send_stub() -> None:
    root = Path(__file__).resolve().parents[1]
    ipc = (root / "gui" / "src" / "types" / "ipc.ts").read_text(encoding="utf-8")
    for state in EXPECTED_STATES:
        assert f"| '{state}'" in ipc or f"= '{state}'" in ipc

    integrations = (
        root / "gui" / "src" / "pages" / "settings" / "integrations" / "IntegrationControls.tsx"
    ).read_text(encoding="utf-8")
    email = (root / "gui" / "src" / "pages" / "EmailPage.tsx").read_text(encoding="utf-8")
    assert "Configured only" in integrations
    assert "Copy draft" in email
    assert "Sending is unavailable in this GUI" in email
    assert "[Email stub]" not in email
