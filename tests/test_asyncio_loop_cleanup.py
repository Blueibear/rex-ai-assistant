from __future__ import annotations

import asyncio
from pathlib import Path

from rex.openclaw import tool_executor


def test_rex_source_has_no_deprecated_get_event_loop_calls() -> None:
    offenders = []
    for path in Path("rex").rglob("*.py"):
        if "asyncio.get_event_loop(" in path.read_text(encoding="utf-8"):
            offenders.append(str(path))
    assert offenders == []


def test_sync_weather_executor_works_when_called_inside_running_loop(monkeypatch) -> None:
    async def fake_get_weather(location: str, api_key: str):
        await asyncio.sleep(0)
        return {"city": location, "temp_f": 75.0, "description": "clear"}

    monkeypatch.setattr("rex.weather.get_weather", fake_get_weather)
    monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "test-key")
    monkeypatch.setattr("rex.credentials.legacy_plaintext_fallback_enabled", lambda: True)

    async def invoke():
        return tool_executor._execute_weather_now({"location": "Dallas"}, {})

    result = asyncio.run(invoke())
    assert result["city"] == "Dallas"
    assert result["temp_f"] == 75.0
    assert "error" not in result
