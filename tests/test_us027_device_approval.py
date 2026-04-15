"""Tests for US-027: device approval and rename workflow.

Covers:
- approve_device() writes alias to device_aliases.json
- approve_device() renames an existing alias
- ignore_device() writes entity_id to device_ignore.json
- load_ignored_devices() reads ignore list
- rex ha approve CLI (mocked interactive path)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rex.ha.discovery import approve_device, ignore_device, load_ignored_devices

# rex.cli raises SystemExit on Python != 3.11; skip CLI tests on other versions
_CLI_AVAILABLE = sys.version_info[:2] == (3, 11)


# ---------------------------------------------------------------------------
# approve_device
# ---------------------------------------------------------------------------


def test_approve_device_creates_alias(tmp_path: Path) -> None:
    """approve_device writes alias to device_aliases.json."""
    aliases_path = tmp_path / "device_aliases.json"
    approve_device("light.bedroom_main", "bedroom light", aliases_path=aliases_path)

    data = json.loads(aliases_path.read_text())
    assert data["aliases"]["bedroom light"] == "light.bedroom_main"


def test_approve_device_normalises_alias_to_lowercase(tmp_path: Path) -> None:
    """approve_device lower-cases the alias key."""
    aliases_path = tmp_path / "device_aliases.json"
    approve_device("switch.fan", "Ceiling Fan", aliases_path=aliases_path)

    data = json.loads(aliases_path.read_text())
    assert "ceiling fan" in data["aliases"]
    assert data["aliases"]["ceiling fan"] == "switch.fan"


def test_approve_device_renames_existing_alias(tmp_path: Path) -> None:
    """approve_device updates an existing alias entry."""
    aliases_path = tmp_path / "device_aliases.json"
    aliases_path.write_text(
        json.dumps({"aliases": {"old name": "light.bedroom_main"}, "synonyms": {}}),
        encoding="utf-8",
    )
    approve_device("light.bedroom_main", "new name", aliases_path=aliases_path)

    data = json.loads(aliases_path.read_text())
    assert data["aliases"]["new name"] == "light.bedroom_main"


def test_approve_device_preserves_existing_synonyms(tmp_path: Path) -> None:
    """approve_device does not clobber existing synonyms."""
    aliases_path = tmp_path / "device_aliases.json"
    aliases_path.write_text(
        json.dumps({"aliases": {}, "synonyms": {"lamp": "light"}}),
        encoding="utf-8",
    )
    approve_device("light.kitchen", "kitchen light", aliases_path=aliases_path)

    data = json.loads(aliases_path.read_text())
    assert data["synonyms"]["lamp"] == "light"


def test_approve_device_creates_parent_dirs(tmp_path: Path) -> None:
    """approve_device creates any missing parent directories."""
    aliases_path = tmp_path / "deep" / "nested" / "aliases.json"
    approve_device("light.x", "x light", aliases_path=aliases_path)
    assert aliases_path.exists()


# ---------------------------------------------------------------------------
# ignore_device / load_ignored_devices
# ---------------------------------------------------------------------------


def test_ignore_device_creates_ignore_file(tmp_path: Path) -> None:
    """ignore_device writes entity_id to device_ignore.json."""
    ignore_path = tmp_path / "device_ignore.json"
    ignore_device("light.old_lamp", ignore_path=ignore_path)

    data = json.loads(ignore_path.read_text())
    assert "light.old_lamp" in data["ignored"]


def test_ignore_device_does_not_duplicate(tmp_path: Path) -> None:
    """ignore_device does not add duplicates."""
    ignore_path = tmp_path / "device_ignore.json"
    ignore_device("light.old_lamp", ignore_path=ignore_path)
    ignore_device("light.old_lamp", ignore_path=ignore_path)

    data = json.loads(ignore_path.read_text())
    assert data["ignored"].count("light.old_lamp") == 1


def test_load_ignored_devices_missing_file(tmp_path: Path) -> None:
    """load_ignored_devices returns [] when file does not exist."""
    result = load_ignored_devices(tmp_path / "no_such_file.json")
    assert result == []


def test_load_ignored_devices_round_trip(tmp_path: Path) -> None:
    """load_ignored_devices reads back what ignore_device wrote."""
    ignore_path = tmp_path / "device_ignore.json"
    ignore_device("sensor.door", ignore_path=ignore_path)
    ignore_device("sensor.window", ignore_path=ignore_path)

    result = load_ignored_devices(ignore_path)
    assert "sensor.door" in result
    assert "sensor.window" in result


# ---------------------------------------------------------------------------
# CLI: rex ha approve (non-interactive, mocked)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CLI_AVAILABLE, reason="rex.cli requires Python 3.11")
def _run_ha_approve(
    devices: list[dict],
    user_inputs: list[str],
    *,
    aliases_path: str | None = None,
    ignore_path: str | None = None,
    ha_configured: bool = True,
) -> int:
    """Drive _cmd_ha_approve with canned inputs and mocked discover_devices."""
    import argparse

    from rex.cli import cmd_ha  # noqa: PLC0415

    args = argparse.Namespace(
        ha_command="approve",
        aliases_path=aliases_path,
        ignore_path=ignore_path,
    )

    cfg_mock = MagicMock()
    cfg_mock.ha_base_url = "http://ha.local" if ha_configured else ""
    cfg_mock.ha_token = "token" if ha_configured else ""

    with (
        patch("rex.config.load_config", return_value=cfg_mock),
        patch("rex.ha.discovery.discover_devices", return_value=devices),
        patch("builtins.input", side_effect=user_inputs),
    ):
        return cmd_ha(args)


@pytest.mark.skipif(not _CLI_AVAILABLE, reason="rex.cli requires Python 3.11")
def test_ha_approve_not_configured() -> None:
    """rex ha approve exits 1 when HA is not configured."""
    rc = _run_ha_approve([], [], ha_configured=False)
    assert rc == 1


@pytest.mark.skipif(not _CLI_AVAILABLE, reason="rex.cli requires Python 3.11")
def test_ha_approve_no_devices(tmp_path: Path) -> None:
    """rex ha approve exits 0 gracefully when no devices are discovered."""
    rc = _run_ha_approve(
        devices=[],
        user_inputs=[],
        aliases_path=str(tmp_path / "aliases.json"),
        ignore_path=str(tmp_path / "ignore.json"),
    )
    assert rc == 0


@pytest.mark.skipif(not _CLI_AVAILABLE, reason="rex.cli requires Python 3.11")
def test_ha_approve_approve_device(tmp_path: Path) -> None:
    """rex ha approve writes approved alias when user enters a name."""
    aliases_path = tmp_path / "aliases.json"
    ignore_path = tmp_path / "ignore.json"
    devices = [
        {
            "entity_id": "light.hall",
            "friendly_name": "Hall Light",
            "state": "off",
            "domain": "light",
        }
    ]

    _run_ha_approve(
        devices=devices,
        user_inputs=["hall light"],
        aliases_path=str(aliases_path),
        ignore_path=str(ignore_path),
    )

    data = json.loads(aliases_path.read_text())
    assert data["aliases"]["hall light"] == "light.hall"


@pytest.mark.skipif(not _CLI_AVAILABLE, reason="rex.cli requires Python 3.11")
def test_ha_approve_ignore_device(tmp_path: Path) -> None:
    """rex ha approve writes ignored entity when user enters 'i'."""
    aliases_path = tmp_path / "aliases.json"
    ignore_path = tmp_path / "ignore.json"
    devices = [
        {
            "entity_id": "sensor.temp",
            "friendly_name": "Temperature",
            "state": "21",
            "domain": "sensor",
        }
    ]

    _run_ha_approve(
        devices=devices,
        user_inputs=["i"],
        aliases_path=str(aliases_path),
        ignore_path=str(ignore_path),
    )

    ignored = load_ignored_devices(ignore_path)
    assert "sensor.temp" in ignored


@pytest.mark.skipif(not _CLI_AVAILABLE, reason="rex.cli requires Python 3.11")
def test_ha_approve_skip_device(tmp_path: Path) -> None:
    """rex ha approve skips a device when user presses Enter."""
    aliases_path = tmp_path / "aliases.json"
    ignore_path = tmp_path / "ignore.json"
    devices = [
        {
            "entity_id": "light.spare",
            "friendly_name": "Spare Light",
            "state": "off",
            "domain": "light",
        }
    ]

    _run_ha_approve(
        devices=devices,
        user_inputs=["s"],
        aliases_path=str(aliases_path),
        ignore_path=str(ignore_path),
    )

    # Nothing should be written
    assert (
        not aliases_path.exists() or json.loads(aliases_path.read_text()).get("aliases", {}) == {}
    )
    assert load_ignored_devices(ignore_path) == []
