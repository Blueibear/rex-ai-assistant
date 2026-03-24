"""Configuration model for the Home Assistant TTS notification channel.

Config path: ``notifications.ha_tts`` in ``config/rex_config.json``.

Keys
----
enabled
    ``false`` by default.  The channel is a no-op unless ``true``.
base_url
    HTTPS URL of the Home Assistant instance
    (e.g. ``"https://homeassistant.local:8123"``).
    HTTP is accepted only for development (see ``allow_http``).
    Must not contain embedded credentials.  Host must not resolve to a
    loopback, private, link-local, reserved, multicast, or unspecified
    address (SSRF defence).
token_ref
    ``CredentialManager`` lookup key for the long-lived access token.
    The resolved value is sent as ``Authorization: Bearer <token>``.
    **Never** store the token directly in ``rex_config.json``.
default_entity_id
    Default target entity (e.g. ``"media_player.living_room"``).
    Required when no per-notification override is supplied.
default_tts_domain
    TTS domain to call (default: ``"tts"``).
default_tts_service
    TTS service within the domain (default: ``"speak"``).
timeout_seconds
    HTTP request timeout (default: ``10.0``).
allow_http
    When ``true`` the SSRF host check is still enforced but ``http://``
    URLs are accepted.  Keep ``false`` in production.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class HaTtsConfig(BaseModel):
    """Parsed ``notifications.ha_tts`` configuration block."""

    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = False
    base_url: str | None = None
    token_ref: str | None = None
    default_entity_id: str | None = None
    default_tts_domain: str = "tts"
    default_tts_service: str = "speak"
    timeout_seconds: float = 10.0
    allow_http: bool = False


def load_ha_tts_config() -> HaTtsConfig:
    """Return the HA TTS config parsed from the active runtime config."""
    try:
        from rex.config_manager import load_config

        cfg = load_config()
        raw = cfg.get("notifications", {}).get("ha_tts", {})
        return HaTtsConfig.model_validate(raw)
    except Exception:
        return HaTtsConfig()
