"""IP geolocation fallback for default location detection.

Uses the free ip-api.com service to detect the user's approximate location
when no default_location is configured. Results are cached in memory for
the session to avoid repeated network calls.
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import Dict, Optional

from rex.logging_utils import get_logger

LOGGER = get_logger(__name__)

_IP_API_URL = "http://ip-api.com/json/?fields=city,timezone,lat,lon,status,message"
_TIMEOUT_SECONDS = 3.0

_location_cache: Optional[Dict[str, object]] = None


async def detect_location() -> Optional[Dict[str, object]]:
    """Detect approximate location via IP geolocation.

    Returns:
        Dict with keys ``city`` (str), ``timezone`` (str), ``lat`` (float),
        ``lon`` (float), or ``None`` if detection fails or times out.

    Notes:
        Result is cached in memory for the session.  If ``default_location``
        is set in config, callers should skip this function entirely.
    """
    global _location_cache
    if _location_cache is not None:
        return _location_cache

    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _fetch_location),
            timeout=_TIMEOUT_SECONDS,
        )
        if result is not None:
            _location_cache = result
        return result
    except asyncio.TimeoutError:
        LOGGER.warning("IP geolocation timed out after %.1fs", _TIMEOUT_SECONDS)
        return None
    except Exception as exc:
        LOGGER.warning("IP geolocation failed: %s", exc)
        return None


def _fetch_location() -> Optional[Dict[str, object]]:
    """Blocking HTTP fetch of ip-api.com.  Run in an executor."""
    try:
        req = urllib.request.Request(
            _IP_API_URL,
            headers={"User-Agent": "Rex-AI-Assistant/1.0"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        if data.get("status") != "success":
            LOGGER.warning("ip-api returned non-success status: %s", data.get("message"))
            return None
        return {
            "city": str(data["city"]),
            "timezone": str(data["timezone"]),
            "lat": float(data["lat"]),
            "lon": float(data["lon"]),
        }
    except Exception as exc:
        LOGGER.warning("_fetch_location error: %s", exc)
        return None


def get_cached_timezone() -> Optional[str]:
    """Return the timezone from the in-memory cache, or None if not available."""
    if _location_cache is not None:
        tz = _location_cache.get("timezone")
        if isinstance(tz, str) and tz:
            return tz
    return None


def get_cached_city() -> Optional[str]:
    """Return the city name from the in-memory cache, or None if not available."""
    if _location_cache is not None:
        city = _location_cache.get("city")
        if isinstance(city, str) and city:
            return city
    return None


def clear_cache() -> None:
    """Clear the in-memory location cache (useful for testing)."""
    global _location_cache
    _location_cache = None
