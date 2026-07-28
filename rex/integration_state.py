"""Truthful, evidence-based integration readiness shared by CLI and APIs."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum


class IntegrationState(StrEnum):
    UNAVAILABLE = "unavailable"
    UNCONFIGURED = "unconfigured"
    CONFIGURED = "configured"
    REACHABLE = "reachable"
    AUTHENTICATED = "authenticated"
    DEGRADED = "degraded"
    READ_ONLY = "read_only"
    WRITE_CAPABLE = "write_capable"
    WRITE_TESTED = "write_tested"
    VERIFIED = "verified"


READ_CAPABLE_STATES = {
    IntegrationState.AUTHENTICATED,
    IntegrationState.READ_ONLY,
    IntegrationState.WRITE_CAPABLE,
    IntegrationState.WRITE_TESTED,
    IntegrationState.VERIFIED,
}
WRITE_CAPABLE_STATES = {
    IntegrationState.WRITE_CAPABLE,
    IntegrationState.WRITE_TESTED,
    IntegrationState.VERIFIED,
}


@dataclass(frozen=True)
class IntegrationEvidence:
    key: str
    name: str
    state: IntegrationState
    configured: bool
    available: bool = True
    testable: bool = False
    configure_url: str = ""
    detail: str = ""

    @property
    def read_capable(self) -> bool:
        return self.state in READ_CAPABLE_STATES

    @property
    def write_capable(self) -> bool:
        return self.state in WRITE_CAPABLE_STATES

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["state"] = self.state.value
        result["read_capable"] = self.read_capable
        result["write_capable"] = self.write_capable
        return result


def configured_state(configured: bool, *, available: bool = True) -> IntegrationState:
    if not available:
        return IntegrationState.UNAVAILABLE
    return IntegrationState.CONFIGURED if configured else IntegrationState.UNCONFIGURED


def build_integration_inventory(
    config: object,
    environ: Mapping[str, str] | None = None,
) -> list[IntegrationEvidence]:
    """Describe configuration evidence without inventing live connectivity."""
    env = os.environ if environ is None else environ

    def has(name: str) -> bool:
        return bool(getattr(config, name, None))

    integrations_config = getattr(config, "integrations", None)
    openclaw_gateway_url = (
        getattr(integrations_config, "openclaw_gateway_url", "")
        if integrations_config is not None
        else getattr(config, "__dict__", {}).get("openclaw_gateway_url", "")
    )

    email_provider = str(getattr(config, "email_provider", "none") or "none")
    calendar_provider = str(getattr(config, "calendar_provider", "none") or "none")
    email_configured = email_provider not in {"", "none"} or bool(
        getattr(config, "email_accounts", [])
    )
    calendar_configured = calendar_provider not in {"", "none"}
    sms_configured = all(
        env.get(key) for key in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER")
    )
    phone_configured = all(
        env.get(key) for key in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER")
    )
    search_configured = bool(
        env.get("SERPAPI_API_KEY")
        or env.get("BRAVE_API_KEY")
        or env.get("GOOGLE_CSE_ID")
        or getattr(config, "search_providers", None)
    )
    specs = [
        (
            "home_assistant",
            "Home Assistant",
            has("ha_base_url") and has("ha_token"),
            True,
            True,
            "/settings/home-assistant",
            "",
        ),
        (
            "email",
            "Email",
            email_configured,
            email_provider != "outlook",
            False,
            "/settings?section=integrations",
            "Outlook Graph OAuth is unavailable" if email_provider == "outlook" else "",
        ),
        (
            "calendar",
            "Calendar",
            calendar_configured,
            calendar_provider != "outlook",
            False,
            "/settings?section=integrations",
            "Outlook Graph OAuth is unavailable" if calendar_provider == "outlook" else "",
        ),
        (
            "sms",
            "SMS (Twilio)",
            sms_configured,
            True,
            False,
            "/settings?section=integrations",
            "Sending is not live-tested by status checks",
        ),
        (
            "phone",
            "Phone (Twilio)",
            phone_configured,
            True,
            False,
            "/settings?section=integrations",
            "Calling is not live-tested by status checks",
        ),
        (
            "telegram",
            "Telegram",
            has("telegram_bot_token") and has("telegram_chat_id"),
            True,
            False,
            "/settings?section=integrations",
            "",
        ),
        (
            "search",
            "Web Search",
            search_configured,
            True,
            False,
            "/settings?section=ai",
            "Configuration does not prove current provider reachability",
        ),
        (
            "mqtt",
            "MQTT",
            bool(env.get("MQTT_BROKER_HOST")),
            True,
            False,
            "/settings?section=integrations",
            "",
        ),
        ("openai", "OpenAI", has("openai_api_key"), True, False, "/settings?section=ai", ""),
        ("ollama", "Ollama", has("ollama_base_url"), True, False, "/settings?section=ai", ""),
        (
            "push",
            "Push Notifications",
            has("push_provider") and has("push_token"),
            True,
            False,
            "/settings?section=integrations",
            "Delivery is not live-tested by status checks",
        ),
        (
            "openclaw",
            "OpenClaw",
            bool(openclaw_gateway_url),
            True,
            False,
            "/settings?section=ai",
            "Optional experimental gateway",
        ),
    ]
    return [
        IntegrationEvidence(
            key=key,
            name=name,
            configured=configured,
            available=available,
            testable=testable,
            configure_url=configure_url,
            state=configured_state(configured, available=available),
            detail=detail,
        )
        for key, name, configured, available, testable, configure_url, detail in specs
    ]


__all__ = [
    "IntegrationEvidence",
    "IntegrationState",
    "READ_CAPABLE_STATES",
    "WRITE_CAPABLE_STATES",
    "build_integration_inventory",
    "configured_state",
]
