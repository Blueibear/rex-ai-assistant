"""Home Assistant TTS client for the Rex notification channel (Cycle 8.3a).

This module provides a minimal HTTP client that calls the Home Assistant
REST API to announce text over a target media player entity.

API used
--------
``POST /api/services/{tts_domain}/{tts_service}``

Payload::

    {
        "entity_id": "<media_player entity>",
        "message": "<text to speak>",
        ...extra_data
    }

Authentication
--------------
Every request carries ``Authorization: Bearer <token>`` resolved via
``CredentialManager`` from the configured ``token_ref`` key.  The token
is **never** logged.

SSRF hardening
--------------
``base_url`` is validated before any network call:

- Scheme must be ``https`` (or ``http`` when ``allow_http=True`` for dev).
- Embedded credentials in the URL are rejected.
- The hostname is resolved via ``socket.getaddrinfo``; loopback, private,
  link-local, reserved, multicast, and unspecified addresses are rejected.

Dependencies
------------
Uses ``requests`` (already in ``[project.dependencies]``).
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

try:
    import requests as _imported_requests
except ImportError:  # pragma: no cover
    _requests: Any | None = None
else:
    _requests = _imported_requests


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TtsResult:
    """Result of a single TTS announcement attempt."""

    ok: bool
    error: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# SSRF validation helpers (mirrors rex/woocommerce/client.py pattern)
# ─────────────────────────────────────────────────────────────────────────────


def _validate_remote_host(hostname: str | None) -> None:
    """Reject localhost/private/reserved targets to reduce SSRF risk."""
    if not hostname:
        raise ValueError("base_url is missing a hostname")

    lowered = hostname.strip().lower()
    if lowered in {"localhost", "localhost.localdomain"}:
        raise ValueError("base_url host must not be localhost or local network")

    try:
        addresses = {ai[4][0] for ai in socket.getaddrinfo(hostname, None)}
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve base_url host: {hostname}") from exc

    for addr in addresses:
        ip = ip_address(addr)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError("base_url host resolves to a local or reserved address")


def _validate_base_url(base_url: str, *, allow_http: bool = False) -> str:
    """Validate and normalise *base_url* for safe HA REST API calls.

    Args:
        base_url: URL to validate.
        allow_http: When ``True`` both ``http`` and ``https`` are accepted.
                    Should only be ``True`` during local development.

    Returns:
        Normalised URL with trailing slash stripped.

    Raises:
        ValueError: If the URL fails any security check.
    """
    parsed = urlparse(base_url)
    scheme = (parsed.scheme or "").lower()
    allowed_schemes = {"http", "https"} if allow_http else {"https"}
    if scheme not in allowed_schemes:
        if allow_http:
            raise ValueError(f"base_url must use http or https scheme, got: {scheme!r}")
        raise ValueError(
            f"base_url must use https scheme (set allow_http=true for local dev), got: {scheme!r}"
        )
    if not parsed.netloc:
        raise ValueError("base_url must include a host")
    if parsed.username or parsed.password:
        raise ValueError("base_url must not include embedded credentials")
    _validate_remote_host(parsed.hostname)
    return base_url.rstrip("/")


# ─────────────────────────────────────────────────────────────────────────────
# Safe error mapper — keeps tokens / full URLs out of logs
# ─────────────────────────────────────────────────────────────────────────────


def _safe_error(exc: Exception) -> str:
    """Return a sanitised error string safe to log or surface to the user."""
    if _requests is not None:
        if isinstance(exc, _requests.Timeout):
            return "HA TTS request timed out"
        if isinstance(exc, _requests.HTTPError):
            code = exc.response.status_code if exc.response is not None else "unknown"
            return f"HA TTS HTTP error (status={code})"
        if isinstance(exc, _requests.ConnectionError):
            return "HA TTS connection error"
        if isinstance(exc, _requests.RequestException):
            return "HA TTS request failed"
    return "HA TTS error"


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────


class HaTtsClient:
    """Send TTS announcements to Home Assistant over its REST API.

    Parameters
    ----------
    base_url:
        HTTPS URL of the Home Assistant instance.
    token:
        Long-lived access token.  **Never** pass via config; resolve from
        ``CredentialManager`` before constructing this client.
    default_entity_id:
        Fallback target entity when callers do not supply one.
    tts_domain:
        HA service domain for TTS (default ``"tts"``).
    tts_service:
        HA service within *tts_domain* (default ``"speak"``).
    timeout:
        Request timeout in seconds (default ``10.0``).
    allow_http:
        Accept ``http://`` base URLs.  Keep ``False`` in production.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        default_entity_id: str | None = None,
        tts_domain: str = "tts",
        tts_service: str = "speak",
        timeout: float = 10.0,
        allow_http: bool = False,
    ) -> None:
        if _requests is None:
            raise RuntimeError(
                "The 'requests' package is required for the HA TTS client. "
                "Install it with: pip install requests"
            )
        self._base_url = _validate_base_url(base_url, allow_http=allow_http)
        # Keep token in a private attribute; never log it.
        self._token = token
        self.default_entity_id = default_entity_id
        self.tts_domain = tts_domain
        self.tts_service = tts_service
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(
        self,
        message: str,
        *,
        entity_id: str | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> TtsResult:
        """Announce *message* on the target Home Assistant entity.

        Args:
            message: Text to synthesise and play.
            entity_id: Override the default target entity.  Required when
                ``default_entity_id`` was not set on construction.
            extra_data: Additional key/value pairs merged into the HA service
                payload (e.g. ``{"language": "en"}``, TTS provider options).
                Values are passed through as-is; callers are responsible for
                sanitising any user-supplied content.

        Returns:
            :class:`TtsResult` with ``ok=True`` on success or ``ok=False``
            plus an ``error`` string on failure.  Secrets are never included
            in the error string.
        """
        target = entity_id or self.default_entity_id
        if not target:
            return TtsResult(
                ok=False,
                error="No entity_id provided and no default_entity_id configured",
            )

        if not message or not message.strip():
            return TtsResult(ok=False, error="message must not be empty")

        path = f"/api/services/{self.tts_domain}/{self.tts_service}"
        payload: dict[str, Any] = {"entity_id": target, "message": message}
        if extra_data:
            payload.update(extra_data)

        requests_module = _requests
        if requests_module is None:
            return TtsResult(ok=False, error="requests is not installed")

        try:
            resp = requests_module.post(
                self._base_url + path,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            logger.info(
                "[HA_TTS] Announced on %s (domain=%s service=%s)",
                target,
                self.tts_domain,
                self.tts_service,
            )
            return TtsResult(ok=True)
        except Exception as exc:
            safe = _safe_error(exc)
            logger.warning("[HA_TTS] Failed to announce: %s", safe)
            return TtsResult(ok=False, error=safe)


# ─────────────────────────────────────────────────────────────────────────────
# Factory — builds a client from runtime config + CredentialManager
# ─────────────────────────────────────────────────────────────────────────────


def build_ha_tts_client() -> HaTtsClient | None:
    """Build a :class:`HaTtsClient` from the active runtime config.

    Returns ``None`` when HA TTS is disabled or not fully configured.
    Credentials are resolved via ``CredentialManager``; the token is never
    stored in the config file.
    """
    from rex.ha_tts.config import load_ha_tts_config

    cfg = load_ha_tts_config()
    if not cfg.enabled:
        return None

    if not cfg.base_url:
        logger.warning("[HA_TTS] notifications.ha_tts.base_url is required when enabled=true")
        return None

    if not cfg.token_ref:
        logger.warning("[HA_TTS] notifications.ha_tts.token_ref is required when enabled=true")
        return None

    token: str | None = None
    try:
        from rex.credentials import CredentialManager

        token = CredentialManager().get_token(cfg.token_ref)
    except Exception as exc:
        logger.warning("[HA_TTS] Could not resolve token from token_ref: %s", exc)

    if not token:
        logger.warning(
            "[HA_TTS] Token resolved to empty string from ref %r; channel disabled",
            cfg.token_ref,
        )
        return None

    try:
        return HaTtsClient(
            base_url=cfg.base_url,
            token=token,
            default_entity_id=cfg.default_entity_id,
            tts_domain=cfg.default_tts_domain,
            tts_service=cfg.default_tts_service,
            timeout=cfg.timeout_seconds,
            allow_http=cfg.allow_http,
        )
    except ValueError as exc:
        logger.warning("[HA_TTS] Invalid configuration: %s", exc)
        return None
