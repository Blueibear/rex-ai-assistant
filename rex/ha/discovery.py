"""Home Assistant device discovery via the /api/states endpoint.

Fetches all entities from a running Home Assistant instance and returns
them as plain dicts.  Results are cached in-process for a configurable
number of seconds (default 300 s / 5 min) so repeated calls within one
session do not hammer the HA API.

If HA is not configured (``ha_base_url`` or ``ha_token`` not set), the
module returns an empty list and logs a warning so the rest of the system
degrades gracefully.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

DeviceInfo = dict  # {entity_id, friendly_name, domain, state}

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_CACHE_TTL_DEFAULT: float = 300.0  # 5 minutes

_cache_entries: list[DeviceInfo] = []
_cache_timestamp: float = 0.0
_cache_ttl: float = _CACHE_TTL_DEFAULT


def set_cache_ttl(seconds: float) -> None:
    """Override the cache TTL (useful for tests or runtime reconfiguration).

    Args:
        seconds: New TTL in seconds.  Pass ``0`` to disable caching.
    """
    global _cache_ttl
    _cache_ttl = seconds


def _cache_is_valid() -> bool:
    if _cache_ttl <= 0:
        return False
    return bool(_cache_entries) and (time.monotonic() - _cache_timestamp) < _cache_ttl


def _set_cache(entries: list[DeviceInfo]) -> None:
    global _cache_entries, _cache_timestamp
    _cache_entries = entries
    _cache_timestamp = time.monotonic()


def clear_cache() -> None:
    """Invalidate the discovery cache (useful for tests)."""
    global _cache_entries, _cache_timestamp
    _cache_entries = []
    _cache_timestamp = 0.0


# ---------------------------------------------------------------------------
# HA API helpers
# ---------------------------------------------------------------------------


def _fetch_states(base_url: str, token: str, verify_ssl: bool, timeout: float) -> list[dict]:
    """Call GET /api/states and return the raw JSON list."""
    import json
    import ssl
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/api/states"
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

    with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx) as resp:
        body = resp.read().decode()
    return json.loads(body)  # type: ignore[no-any-return]


def _parse_entity(raw: dict) -> DeviceInfo:
    """Convert a raw HA state dict to our canonical DeviceInfo shape."""
    entity_id: str = raw.get("entity_id", "")
    domain: str = entity_id.split(".")[0] if "." in entity_id else ""
    attributes: dict = raw.get("attributes", {})
    friendly_name: str = attributes.get("friendly_name", entity_id)
    state: str = str(raw.get("state", "unknown"))
    return {
        "entity_id": entity_id,
        "friendly_name": friendly_name,
        "domain": domain,
        "state": state,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def discover_devices(
    base_url: str | None = None,
    token: str | None = None,
    *,
    verify_ssl: bool = True,
    timeout: float = 10.0,
    cache_ttl: float | None = None,
) -> list[DeviceInfo]:
    """Return all HA entities as a list of :data:`DeviceInfo` dicts.

    Results are cached for *cache_ttl* seconds (default ``_CACHE_TTL_DEFAULT``).
    If HA is not configured, returns ``[]`` and logs a warning.

    Args:
        base_url: Base URL of the Home Assistant instance.  Falls back to
            the global module cache if already populated.
        token: Long-lived access token.
        verify_ssl: Whether to verify the HA server's TLS certificate.
        timeout: HTTP request timeout in seconds.
        cache_ttl: Override the module-level TTL for this call only.
            ``None`` uses the current module TTL.

    Returns:
        List of dicts with keys ``entity_id``, ``friendly_name``,
        ``domain``, and ``state``.
    """
    if cache_ttl is not None:
        set_cache_ttl(cache_ttl)

    if _cache_is_valid():
        logger.debug("HA discovery: returning %d cached entities", len(_cache_entries))
        return list(_cache_entries)

    if not base_url or not token:
        logger.warning(
            "HA discovery: Home Assistant not configured "
            "(set ha_base_url and HA_TOKEN); returning empty list"
        )
        return []

    try:
        raw_states = _fetch_states(base_url, token, verify_ssl=verify_ssl, timeout=timeout)
    except Exception as exc:
        logger.error("HA discovery: failed to fetch /api/states: %s", exc)
        return []

    entries = [_parse_entity(s) for s in raw_states]
    _set_cache(entries)
    logger.info(
        "HA discovery: found %d entities",
        len(entries),
        extra={"event": "ha_discovery_complete"},
    )
    return list(entries)
