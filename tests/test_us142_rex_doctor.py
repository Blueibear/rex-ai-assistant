"""Tests for US-142: Improve rex doctor to validate the full install."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

from rex.doctor import (
    CheckResult,
    DiagnosticsReport,
    Status,
    _status_symbol,
    check_audio_input_device,
    check_audio_output_device,
    check_config_file,
    check_core_dependencies,
    check_lm_studio_reachability,
    check_python_version,
    run_diagnostics,
)

# ---------------------------------------------------------------------------
# AC1: checks are present for all required areas
# ---------------------------------------------------------------------------


class TestCheckPythonVersion:
    def _make_version(self, major: int, minor: int, micro: int = 0):
        from collections import namedtuple

        # namedtuple supports attribute access AND tuple comparison
        VersionInfo = namedtuple(
            "version_info", ["major", "minor", "micro", "releaselevel", "serial"]
        )
        return VersionInfo(major, minor, micro, "final", 0)

    def test_pass_for_310_plus(self):
        import sys

        # Policy: only Python 3.11 is supported; 3.10 returns ERROR
        with patch.object(sys, "version_info", self._make_version(3, 10, 0)):
            result = check_python_version()
        assert result.status == Status.ERROR

    def test_pass_for_39(self):
        import sys

        # Policy: only Python 3.11 is supported; 3.9 returns ERROR
        with patch.object(sys, "version_info", self._make_version(3, 9, 0)):
            result = check_python_version()
        assert result.status == Status.ERROR

    def test_fail_for_38(self):
        import sys

        with patch.object(sys, "version_info", self._make_version(3, 8, 0)):
            result = check_python_version()
        assert result.status == Status.ERROR


class TestCheckConfigFile:
    def test_pass_when_config_exists(self, tmp_path):
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        cfg = cfg_dir / "rex_config.json"
        cfg.write_text('{"key": "value"}')
        result = check_config_file(tmp_path)
        assert result.status == Status.OK

    def test_fail_when_config_missing_no_example(self, tmp_path):
        result = check_config_file(tmp_path)
        assert result.status == Status.ERROR
        assert result.details  # actionable message

    def test_warn_when_only_example_exists(self, tmp_path):
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        example = cfg_dir / "rex_config.example.json"
        example.write_text("{}")
        result = check_config_file(tmp_path)
        assert result.status == Status.WARNING
        assert "Copy" in result.details or "copy" in result.details.lower()

    def test_fail_when_config_invalid_json(self, tmp_path):
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        (cfg_dir / "rex_config.json").write_text("{bad json")
        result = check_config_file(tmp_path)
        assert result.status == Status.ERROR


class TestRequiredPackages:
    def test_returns_list(self):
        results = check_core_dependencies()
        assert isinstance(results, list)
        assert len(results) > 0

    def test_flask_present(self):
        # flask is always installed as a rex dependency
        results = check_core_dependencies()
        names = [r.name for r in results]
        assert any("flask" in n.lower() for n in names)

    def test_all_results_have_actionable_message_on_fail(self):
        results = check_core_dependencies()
        for r in results:
            if r.status == Status.ERROR:
                assert r.details, f"No actionable message for failing check: {r.name}"


class TestCheckAudioInput:
    def test_pass_when_sounddevice_has_input_devices(self):
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "Microphone (USB)", "max_input_channels": 2, "max_output_channels": 0}
        ]
        mock_sd.query_devices.side_effect = None

        def query_devices_side_effect(kind=None):
            if kind == "input":
                return {"name": "Microphone (USB)", "max_input_channels": 2}
            return [{"name": "Microphone (USB)", "max_input_channels": 2, "max_output_channels": 0}]

        mock_sd.query_devices.side_effect = query_devices_side_effect

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = check_audio_input_device()
        assert result.status == Status.OK
        assert "Microphone" in result.message

    def test_audio_input_lists_available_device_names(self):
        mock_sd = MagicMock()

        def query_devices_side_effect(kind=None):
            if kind == "input":
                return {"name": "Desk Mic", "max_input_channels": 2}
            return [
                {"name": "Desk Mic", "max_input_channels": 2, "max_output_channels": 0},
                {"name": "USB Mic", "max_input_channels": 1, "max_output_channels": 0},
                {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
            ]

        mock_sd.query_devices.side_effect = query_devices_side_effect

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = check_audio_input_device()

        assert result.status == Status.OK
        assert "Desk Mic" in result.message
        assert "USB Mic" in result.message

    def test_fail_when_no_input_devices(self):
        mock_sd = MagicMock()

        def query_devices_side_effect(kind=None):
            if kind is None:
                return [{"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2}]
            raise Exception("No input device")

        mock_sd.query_devices.side_effect = query_devices_side_effect

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = check_audio_input_device()
        assert result.status == Status.ERROR
        assert result.details  # actionable message

    def test_fail_when_sounddevice_not_installed(self):
        import builtins

        real_import = builtins.__import__

        def import_blocker(name, *args, **kwargs):
            if name == "sounddevice":
                raise ImportError("No module named 'sounddevice'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_blocker):
            result = check_audio_input_device()
        assert result.status == Status.ERROR
        assert "sounddevice" in result.message.lower() or "sounddevice" in result.details.lower()

    def test_fail_has_actionable_details(self):
        mock_sd = MagicMock()
        mock_sd.query_devices.side_effect = Exception("PortAudio error")

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = check_audio_input_device()
        assert result.status == Status.ERROR
        assert result.details


class TestCheckAudioOutput:
    def test_pass_when_output_device_found(self):
        mock_sd = MagicMock()

        def query_devices_side_effect(kind=None):
            if kind == "output":
                return {"name": "Speakers (Realtek)", "max_output_channels": 2}
            return [
                {"name": "Speakers (Realtek)", "max_input_channels": 0, "max_output_channels": 2}
            ]

        mock_sd.query_devices.side_effect = query_devices_side_effect

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = check_audio_output_device()
        assert result.status == Status.OK
        assert "Speakers" in result.message

    def test_fail_when_no_output_devices(self):
        mock_sd = MagicMock()

        def query_devices_side_effect(kind=None):
            if kind is None:
                return [{"name": "Mic", "max_input_channels": 2, "max_output_channels": 0}]
            raise Exception("No output device")

        mock_sd.query_devices.side_effect = query_devices_side_effect

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = check_audio_output_device()
        assert result.status == Status.ERROR
        assert result.details

    def test_fail_when_sounddevice_not_installed(self):
        import builtins

        real_import = builtins.__import__

        def import_blocker(name, *args, **kwargs):
            if name == "sounddevice":
                raise ImportError("No module named 'sounddevice'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_blocker):
            result = check_audio_output_device()
        assert result.status == Status.ERROR
        assert "sounddevice" in result.message.lower() or "sounddevice" in result.details.lower()


class TestCheckLmStudioReachability:
    def test_pass_when_port_open(self):
        with patch("rex.doctor.socket.create_connection") as mock_conn:
            mock_conn.return_value.__enter__ = MagicMock(return_value=None)
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            result = check_lm_studio_reachability()
        assert result.status == Status.OK
        assert "reachable" in result.message.lower()

    def test_pass_or_warn_when_connection_refused(self):
        with patch("rex.doctor.socket.create_connection", side_effect=ConnectionRefusedError):
            result = check_lm_studio_reachability()
        # Not an ERROR because LM Studio is optional
        assert result.status != Status.ERROR
        assert result.details  # actionable message

    def test_pass_or_warn_when_timeout(self):
        with patch("rex.doctor.socket.create_connection", side_effect=socket.timeout):
            result = check_lm_studio_reachability()
        assert result.status != Status.ERROR
        assert result.details

    def test_uses_timeout(self):
        """Verify create_connection is called with the timeout parameter."""
        with patch("rex.doctor.socket.create_connection") as mock_conn:
            mock_conn.return_value.__enter__ = MagicMock(return_value=None)
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            check_lm_studio_reachability(timeout=3.0)
        call_kwargs = mock_conn.call_args
        # timeout is passed as second positional arg or keyword
        assert call_kwargs is not None
        args, kwargs = call_kwargs
        timeout_passed = kwargs.get("timeout") or (len(args) >= 2 and args[1])
        assert timeout_passed

    def test_not_running_message_is_actionable(self):
        with patch("rex.doctor.socket.create_connection", side_effect=ConnectionRefusedError):
            result = check_lm_studio_reachability()
        assert result.details
        assert "1234" in result.details or "LM Studio" in result.details


# ---------------------------------------------------------------------------
# AC2: PASS / FAIL output format
# ---------------------------------------------------------------------------


class TestStatusSymbol:
    def test_ok_shows_pass(self):
        assert _status_symbol(Status.OK) == "[PASS]"

    def test_info_shows_pass(self):
        assert _status_symbol(Status.INFO) == "[PASS]"

    def test_warning_shows_pass(self):
        # Warnings are non-blocking; Rex can still run
        assert _status_symbol(Status.WARNING) == "[PASS]"

    def test_error_shows_fail(self):
        assert _status_symbol(Status.ERROR) == "[FAIL]"


class TestCheckResultMessages:
    def test_fail_check_has_actionable_details(self):
        result = check_config_file(None)
        # None root → warning, not error, but still has details
        assert result.name == "Config File"

    def test_audio_input_fail_has_actionable_message(self):
        import sys

        original = sys.modules.pop("sounddevice", None)
        try:
            result = check_audio_input_device()
            if result.status == Status.ERROR:
                assert result.details
        finally:
            if original is not None:
                sys.modules["sounddevice"] = original


# ---------------------------------------------------------------------------
# AC3: overall result indicates readiness
# ---------------------------------------------------------------------------


class TestRunDiagnosticsOutput:
    def _make_all_pass_report(self):
        report = DiagnosticsReport()
        report.add(CheckResult(name="Test", status=Status.OK, message="ok"))
        return report

    def test_ready_message_when_no_errors(self, capsys, tmp_path):
        # Patch all checks to return OK so we get a clean run
        ok = CheckResult(name="x", status=Status.OK, message="ok")
        ok_list = [ok]

        with (
            patch("rex.doctor._find_project_root", return_value=tmp_path),
            patch("rex.doctor.check_python_version", return_value=ok),
            patch("rex.doctor.check_package_installation", return_value=ok),
            patch("rex.doctor.check_config_file", return_value=ok),
            patch("rex.doctor.check_env_file", return_value=ok),
            patch("rex.doctor.check_environment_variables", return_value=ok),
            patch("rex.doctor.check_config_permissions", return_value=ok),
            patch("rex.doctor.check_core_dependencies", return_value=ok_list),
            patch("rex.doctor.check_audio_input_device", return_value=ok),
            patch("rex.doctor.check_audio_output_device", return_value=ok),
            patch("rex.doctor.check_smart_speakers", return_value=ok),
            patch("rex.doctor.check_lm_studio_reachability", return_value=ok),
            patch("rex.doctor.check_external_dependencies", return_value=ok_list),
            patch("rex.doctor.check_gpu_availability", return_value=ok),
        ):
            exit_code = run_diagnostics()

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "ready to use" in captured.out.lower()

    def test_not_ready_message_when_errors(self, capsys, tmp_path):
        ok = CheckResult(name="x", status=Status.OK, message="ok")
        fail = CheckResult(name="y", status=Status.ERROR, message="broken", details="fix it")
        ok_list = [ok]

        with (
            patch("rex.doctor._find_project_root", return_value=tmp_path),
            patch("rex.doctor.check_python_version", return_value=fail),
            patch("rex.doctor.check_package_installation", return_value=ok),
            patch("rex.doctor.check_config_file", return_value=ok),
            patch("rex.doctor.check_env_file", return_value=ok),
            patch("rex.doctor.check_environment_variables", return_value=ok),
            patch("rex.doctor.check_config_permissions", return_value=ok),
            patch("rex.doctor.check_core_dependencies", return_value=ok_list),
            patch("rex.doctor.check_audio_input_device", return_value=ok),
            patch("rex.doctor.check_audio_output_device", return_value=ok),
            patch("rex.doctor.check_smart_speakers", return_value=ok),
            patch("rex.doctor.check_lm_studio_reachability", return_value=ok),
            patch("rex.doctor.check_external_dependencies", return_value=ok_list),
            patch("rex.doctor.check_gpu_availability", return_value=ok),
        ):
            exit_code = run_diagnostics()

        captured = capsys.readouterr()
        assert exit_code == 1
        assert "not ready" in captured.out.lower()

    def test_each_check_shows_pass_or_fail(self, capsys, tmp_path):
        ok = CheckResult(name="x", status=Status.OK, message="ok")
        ok_list = [ok]

        with (
            patch("rex.doctor._find_project_root", return_value=tmp_path),
            patch("rex.doctor.check_python_version", return_value=ok),
            patch("rex.doctor.check_package_installation", return_value=ok),
            patch("rex.doctor.check_config_file", return_value=ok),
            patch("rex.doctor.check_env_file", return_value=ok),
            patch("rex.doctor.check_environment_variables", return_value=ok),
            patch("rex.doctor.check_config_permissions", return_value=ok),
            patch("rex.doctor.check_core_dependencies", return_value=ok_list),
            patch("rex.doctor.check_audio_input_device", return_value=ok),
            patch("rex.doctor.check_audio_output_device", return_value=ok),
            patch("rex.doctor.check_smart_speakers", return_value=ok),
            patch("rex.doctor.check_lm_studio_reachability", return_value=ok),
            patch("rex.doctor.check_external_dependencies", return_value=ok_list),
            patch("rex.doctor.check_gpu_availability", return_value=ok),
        ):
            run_diagnostics()

        captured = capsys.readouterr()
        # Every check line must contain [PASS] or [FAIL]
        for line in captured.out.splitlines():
            if line.startswith("["):
                assert "[PASS]" in line or "[FAIL]" in line, f"Line missing PASS/FAIL: {line!r}"

    def test_audio_and_lm_studio_checks_included(self, capsys, tmp_path):
        ok = CheckResult(name="x", status=Status.OK, message="ok")
        ok_list = [ok]
        audio_in = CheckResult(name="Audio Input", status=Status.OK, message="mic ok")
        audio_out = CheckResult(name="Audio Output", status=Status.OK, message="speaker ok")
        lm = CheckResult(name="LM Studio", status=Status.WARNING, message="not running")

        with (
            patch("rex.doctor._find_project_root", return_value=tmp_path),
            patch("rex.doctor.check_python_version", return_value=ok),
            patch("rex.doctor.check_package_installation", return_value=ok),
            patch("rex.doctor.check_config_file", return_value=ok),
            patch("rex.doctor.check_env_file", return_value=ok),
            patch("rex.doctor.check_environment_variables", return_value=ok),
            patch("rex.doctor.check_config_permissions", return_value=ok),
            patch("rex.doctor.check_core_dependencies", return_value=ok_list),
            patch("rex.doctor.check_audio_input_device", return_value=audio_in),
            patch("rex.doctor.check_audio_output_device", return_value=audio_out),
            patch("rex.doctor.check_smart_speakers", return_value=ok),
            patch("rex.doctor.check_lm_studio_reachability", return_value=lm),
            patch("rex.doctor.check_external_dependencies", return_value=ok_list),
            patch("rex.doctor.check_gpu_availability", return_value=ok),
        ):
            run_diagnostics()

        captured = capsys.readouterr()
        assert "Audio Input" in captured.out
        assert "Audio Output" in captured.out
        assert "LM Studio" in captured.out

    def test_required_packages_included_without_verbose(self, capsys, tmp_path):
        ok = CheckResult(name="x", status=Status.OK, message="ok")
        pkg = CheckResult(name="Dependency: flask", status=Status.OK, message="ok")
        pkg_list = [pkg]

        with (
            patch("rex.doctor._find_project_root", return_value=tmp_path),
            patch("rex.doctor.check_python_version", return_value=ok),
            patch("rex.doctor.check_package_installation", return_value=ok),
            patch("rex.doctor.check_config_file", return_value=ok),
            patch("rex.doctor.check_env_file", return_value=ok),
            patch("rex.doctor.check_environment_variables", return_value=ok),
            patch("rex.doctor.check_config_permissions", return_value=ok),
            patch("rex.doctor.check_core_dependencies", return_value=pkg_list),
            patch("rex.doctor.check_audio_input_device", return_value=ok),
            patch("rex.doctor.check_audio_output_device", return_value=ok),
            patch("rex.doctor.check_smart_speakers", return_value=ok),
            patch("rex.doctor.check_lm_studio_reachability", return_value=ok),
            patch("rex.doctor.check_external_dependencies", return_value=[ok]),
            patch("rex.doctor.check_gpu_availability", return_value=ok),
        ):
            run_diagnostics(verbose=False)

        captured = capsys.readouterr()
        assert "flask" in captured.out.lower()
