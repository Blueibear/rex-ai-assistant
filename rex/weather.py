"""Weather integration using OpenWeatherMap Current Weather API."""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_OWM_BASE = "https://api.openweathermap.org/data/2.5/weather"


async def get_weather(location: str, api_key: str) -> dict[str, Any]:
    """Fetch current weather for *location* from OpenWeatherMap.

    Returns a dict with keys:
        temp_f, temp_c, description, humidity, wind_mph, city

    On failure returns ``{"error": "<reason>"}``.

    Args:
        location: City name or "City, CountryCode" string.
        api_key: OpenWeatherMap API key.
    """
    if not location or not api_key:
        return {"error": "location and api_key are required"}

    url = f"{_OWM_BASE}?q={urllib.parse.quote(location)}" f"&appid={api_key}&units=metric"

    loop = asyncio.get_running_loop()
    try:
        raw = await asyncio.wait_for(
            loop.run_in_executor(None, _fetch_weather, url),
            timeout=5.0,
        )
    except asyncio.TimeoutError:
        return {"error": "Weather API request timed out"}
    except Exception as exc:
        return {"error": f"Weather API request failed: {exc}"}

    if "error" in raw:
        return raw

    try:
        temp_c: float = raw["main"]["temp"]
        temp_f = round(temp_c * 9 / 5 + 32, 1)
        description: str = raw["weather"][0]["description"]
        humidity: int = raw["main"]["humidity"]
        wind_ms: float = raw["wind"]["speed"]
        wind_mph = round(wind_ms * 2.237, 1)
        city: str = raw.get("name", location)
        return {
            "temp_f": temp_f,
            "temp_c": round(temp_c, 1),
            "description": description,
            "humidity": humidity,
            "wind_mph": wind_mph,
            "city": city,
        }
    except (KeyError, IndexError, TypeError) as exc:
        return {"error": f"Unexpected weather API response format: {exc}"}


def _fetch_weather(url: str) -> dict[str, Any]:
    """Blocking HTTP call; runs in a thread executor."""
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            msg = json.loads(body).get("message", str(exc))
        except Exception:
            msg = str(exc)
        return {"error": f"Weather API error {exc.code}: {msg}"}
    except Exception as exc:
        return {"error": f"Weather request failed: {exc}"}


__all__ = ["get_weather"]
