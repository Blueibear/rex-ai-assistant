"""Truthful mobile capability reporting (issue #323, Session 1).

A feature is ``true`` only when its real backend path and automated tests
exist.  Configuration alone never makes a capability true, and the response
must never expose secrets, file paths, model paths, account IDs, usernames,
or integration tokens.

Session 1 implements authentication only; every runtime feature stays false
until Session 2 lands its real implementation.
"""

from __future__ import annotations

from typing import Any

from rex.config import MobileApiConfig

MINIMUM_APP_VERSION = "0.1.0"


def server_version() -> str:
    """Return the installed package version, or a safe placeholder."""
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version("askrex-assistant")
    except Exception:  # pragma: no cover - metadata edge cases
        return "unknown"


def resolve_features() -> dict[str, bool]:
    """Return the truthful feature map for the current build.

    Session 1: authentication is implemented and tested; chat, streaming,
    WebSocket, voice, TTS, live voice, notifications, approvals, and Home
    Assistant are explicit 501 scaffolds and therefore false.
    """
    return {
        "authentication": True,
        "chat": False,
        "chat_streaming": False,
        "websocket_chat": False,
        "voice_upload": False,
        "tts": False,
        "live_voice": False,
        "notifications": False,
        "approvals": False,
        "home_assistant": False,
    }


def capabilities_payload(config: MobileApiConfig) -> dict[str, Any]:
    """Build the ``GET /mobile/capabilities`` response body."""
    return {
        "api_version": config.api_version,
        "minimum_app_version": MINIMUM_APP_VERSION,
        "server_version": server_version(),
        "features": resolve_features(),
    }


__all__ = [
    "MINIMUM_APP_VERSION",
    "capabilities_payload",
    "resolve_features",
    "server_version",
]
