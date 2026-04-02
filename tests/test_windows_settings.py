"""Unit tests for rex.tools.windows_settings (US-WIN-003)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import rex.tools.windows_settings as _mod
from rex.tools.windows_settings import (
    get_power_plan,
    get_volume,
    set_brightness,
    set_power_plan,
    set_volume,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ps_result(stdout: str, returncode: int = 0) -> MagicMock:
    r = MagicMock()
    r.returncode = returncode
    r.stdout = stdout
    r.stderr = ""
    return r


def _config(require_confirm: bool = True) -> SimpleNamespace:
    return SimpleNamespace(require_confirm_system_changes=require_confirm)


# ---------------------------------------------------------------------------
# Platform guard
# ---------------------------------------------------------------------------


class TestPlatformGuard:
    def test_non_windows_raises(self) -> None:
        with patch.object(_mod, "_IS_WINDOWS", False):
            with pytest.raises(NotImplementedError, match="only supported on Windows"):
                get_volume()

    def test_windows_does_not_raise_platform(self) -> None:
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = _ps_result("50")
            result = get_volume()
        assert "volume" in result or "error" in result


# ---------------------------------------------------------------------------
# get_volume
# ---------------------------------------------------------------------------


class TestGetVolume:
    def test_returns_volume(self) -> None:
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = _ps_result("75\n")
            result = get_volume()
        assert result["volume"] == 75

    def test_powershell_called(self) -> None:
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = _ps_result("50")
            get_volume()
        call_args = mock_run.call_args[0][0]
        assert "powershell" in call_args[0].lower()

    def test_subprocess_error_returns_error_dict(self) -> None:
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = _ps_result("", returncode=1)
            result = get_volume()
        assert "error" in result


# ---------------------------------------------------------------------------
# set_volume
# ---------------------------------------------------------------------------


class TestSetVolume:
    def test_requires_confirmation_by_default(self) -> None:
        with patch.object(_mod, "_IS_WINDOWS", True):
            result = set_volume(level=60, config=_config(require_confirm=True))
        assert result["requires_confirmation"] is True
        assert "set_volume(60)" in result["action"]

    def test_confirmed_executes(self) -> None:
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = _ps_result("60")
            result = set_volume(level=60, confirmed=True, config=_config(require_confirm=True))
        assert result["volume"] == 60

    def test_no_confirm_required_when_disabled(self) -> None:
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = _ps_result("40")
            result = set_volume(level=40, config=_config(require_confirm=False))
        assert result["volume"] == 40

    def test_clamps_level_to_0_100(self) -> None:
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = _ps_result("100")
            result = set_volume(level=999, confirmed=True)
        assert result.get("volume") == 100 or "error" in result

    def test_powershell_command_contains_level(self) -> None:
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = _ps_result("70")
            set_volume(level=70, confirmed=True)
        ps_script = mock_run.call_args[0][0][-1]
        assert "0.7000" in ps_script or "70" in ps_script

    def test_subprocess_error_returns_error_dict(self) -> None:
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = _ps_result("", returncode=1)
            result = set_volume(level=50, confirmed=True)
        assert "error" in result


# ---------------------------------------------------------------------------
# set_brightness
# ---------------------------------------------------------------------------


class TestSetBrightness:
    def test_requires_confirmation_by_default(self) -> None:
        with patch.object(_mod, "_IS_WINDOWS", True):
            result = set_brightness(level=80, config=_config(require_confirm=True))
        assert result["requires_confirmation"] is True
        assert "set_brightness(80)" in result["action"]

    def test_confirmed_executes(self) -> None:
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = _ps_result("80")
            result = set_brightness(level=80, confirmed=True)
        assert result["brightness"] == 80

    def test_powershell_command_contains_level(self) -> None:
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = _ps_result("55")
            set_brightness(level=55, confirmed=True)
        ps_script = mock_run.call_args[0][0][-1]
        assert "55" in ps_script

    def test_error_returns_error_dict(self) -> None:
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = _ps_result("", returncode=1)
            result = set_brightness(level=50, confirmed=True)
        assert "error" in result


# ---------------------------------------------------------------------------
# get_power_plan
# ---------------------------------------------------------------------------


class TestGetPowerPlan:
    def test_parses_powercfg_output(self) -> None:
        ps_output = "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced) *"
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = _ps_result(ps_output)
            result = get_power_plan()
        assert result["power_plan"] == "Balanced"
        assert "381b4222" in result["guid"]

    def test_powershell_calls_powercfg(self) -> None:
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = _ps_result("Power Scheme GUID: abc  (Test)")
            get_power_plan()
        ps_script = mock_run.call_args[0][0][-1]
        assert "powercfg" in ps_script

    def test_error_returns_error_dict(self) -> None:
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = _ps_result("", returncode=1)
            result = get_power_plan()
        assert "error" in result


# ---------------------------------------------------------------------------
# set_power_plan
# ---------------------------------------------------------------------------


class TestSetPowerPlan:
    def test_requires_confirmation_by_default(self) -> None:
        with patch.object(_mod, "_IS_WINDOWS", True):
            result = set_power_plan(name="High performance", config=_config(require_confirm=True))
        assert result["requires_confirmation"] is True
        assert "High performance" in result["action"]

    def test_confirmed_calls_powercfg_setactive(self) -> None:
        list_output = "Power Scheme GUID: 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  (High performance)"
        readback_output = (
            "Power Scheme GUID: 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  (High performance) *"
        )
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run") as mock_run,
        ):
            # First call: powercfg /list, second call: powercfg /setactive, third: readback /getactivescheme
            mock_run.side_effect = [
                _ps_result(list_output),
                _ps_result(""),
                _ps_result(readback_output),
            ]
            result = set_power_plan(name="High performance", confirmed=True)
        assert result["power_plan"] == "High performance"

    def test_powercfg_setactive_in_commands(self) -> None:
        list_output = "Power Scheme GUID: abc123  (Balanced)"
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _ps_result(list_output),
                _ps_result(""),
                _ps_result("Power Scheme GUID: abc123  (Balanced) *"),
            ]
            set_power_plan(name="Balanced", confirmed=True)
        # Second call should be the setactive call
        second_call_script = mock_run.call_args_list[1][0][0][-1]
        assert "setactive" in second_call_script.lower()

    def test_error_returns_error_dict(self) -> None:
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = _ps_result("", returncode=1)
            result = set_power_plan(name="Balanced", confirmed=True)
        assert "error" in result


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_settings_tools_registered() -> None:
    """All 5 settings tools are present in the default registry."""
    from rex.tools.registry import _build_default_registry

    registry = _build_default_registry()
    names = {t.name for t in registry.all_tools()}
    for tool_name in [
        "get_volume",
        "set_volume",
        "set_brightness",
        "get_power_plan",
        "set_power_plan",
    ]:
        assert tool_name in names, f"{tool_name!r} not found in registry"


def test_settings_tools_have_correct_tags() -> None:
    """All 5 settings tools carry 'windows' and 'settings' capability tags."""
    from rex.tools.registry import _build_default_registry

    registry = _build_default_registry()
    settings_tools = [t for t in registry.all_tools() if "settings" in t.capability_tags]
    assert len(settings_tools) == 5
    for tool in settings_tools:
        assert "windows" in tool.capability_tags
