from __future__ import annotations

import tomllib
from pathlib import Path
from zoneinfo import ZoneInfo

from rex.openclaw.tool_executor import execute_tool


def test_base_dependencies_include_cross_platform_timezone_data() -> None:
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = pyproject["project"]["dependencies"]
    assert any(dependency.startswith("tzdata") for dependency in dependencies)


def test_required_iana_zones_are_available() -> None:
    assert ZoneInfo("America/New_York").key == "America/New_York"
    assert ZoneInfo("America/Chicago").key == "America/Chicago"
    assert ZoneInfo("Europe/London").key == "Europe/London"
    assert ZoneInfo("Asia/Tokyo").key == "Asia/Tokyo"


def test_time_tool_resolves_requested_city_without_default_location_fallback() -> None:
    result = execute_tool(
        {"tool": "time_now", "args": {"location": "New York, NY"}},
        {"location": "Dallas, TX", "timezone": "America/Chicago"},
        skip_policy_check=True,
        skip_credential_check=True,
        skip_audit_log=True,
    )

    assert "error" not in result
    assert result["timezone"] == "America/New_York"
    assert result["local_time"]
