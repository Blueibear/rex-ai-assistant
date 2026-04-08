"""Home Assistant real-time device state queries.

Queries the HA ``/api/states/<entity_id>`` endpoint and returns a structured
dict with the entity's current state and selected attributes.

If HA is not configured, or the entity is not found, the functions return
``None`` and log at the appropriate level so callers degrade gracefully.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

DeviceState = dict  # {entity_id, state, attributes: {brightness, volume, media_title, ...}}


# ---------------------------------------------------------------------------
# HA API helpers
# ---------------------------------------------------------------------------


def _fetch_entity_state(
    base_url: str,
    token: str,
    entity_id: str,
    verify_ssl: bool,
    timeout: float,
) -> dict | None:
    """Call ``GET /api/states/<entity_id>`` and return the raw JSON dict.

    Returns ``None`` when the entity is not found (HTTP 404).
    Raises for other HTTP or network errors.
    """
    import json
    import ssl
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + f"/api/states/{entity_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")

    ssl_ctx: ssl.SSLContext | None = None
    if not verify_ssl:
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx) as resp:
            body = resp.read().decode()
        return json.loads(body)  # type: ignore[no-any-return]
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _parse_state(raw: dict) -> DeviceState:
    """Convert a raw HA state dict to our canonical :data:`DeviceState` shape."""
    entity_id: str = raw.get("entity_id", "")
    state: str = str(raw.get("state", "unknown"))
    raw_attrs: dict = raw.get("attributes", {})

    # Extract the most useful attributes; include the rest verbatim.
    attributes: dict = {
        "brightness": raw_attrs.get("brightness"),
        "volume": raw_attrs.get("volume_level"),
        "media_title": raw_attrs.get("media_title"),
        "friendly_name": raw_attrs.get("friendly_name", entity_id),
    }
    # Merge remaining attributes so callers have full access.
    for key, value in raw_attrs.items():
        if key not in attributes:
            attributes[key] = value

    return {
        "entity_id": entity_id,
        "state": state,
        "attributes": attributes,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_device_state(
    entity_id: str,
    base_url: str | None = None,
    token: str | None = None,
    *,
    verify_ssl: bool = True,
    timeout: float = 10.0,
) -> DeviceState | None:
    """Return current state for *entity_id* from Home Assistant.

    Args:
        entity_id: HA entity ID, e.g. ``light.kitchen_ceiling``.
        base_url: Base URL of the Home Assistant instance.
        token: Long-lived access token.
        verify_ssl: Whether to verify the HA server's TLS certificate.
        timeout: HTTP request timeout in seconds.

    Returns:
        A :data:`DeviceState` dict with keys ``entity_id``, ``state``, and
        ``attributes`` (which always contains ``brightness``, ``volume``,
        ``media_title``, and ``friendly_name``), or ``None`` if the entity
        is not found or HA is not configured.
    """
    if not base_url or not token:
        logger.warning(
            "device_state: Home Assistant not configured; cannot query %s",
            entity_id,
        )
        return None

    try:
        raw = _fetch_entity_state(
            base_url, token, entity_id, verify_ssl=verify_ssl, timeout=timeout
        )
    except Exception as exc:
        logger.error("device_state: failed to query %s: %s", entity_id, exc)
        return None

    if raw is None:
        logger.debug("device_state: entity %r not found in Home Assistant", entity_id)
        return None

    parsed = _parse_state(raw)
    logger.debug(
        "device_state: %s is %r",
        entity_id,
        parsed["state"],
        extra={"event": "ha_device_state_queried"},
    )
    return parsed
