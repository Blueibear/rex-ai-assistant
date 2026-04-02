"""Unit tests for rex.tools.windows_diagnostics (US-WIN-002)."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from rex.tools.windows_diagnostics import (
    get_battery_status,
    get_cpu_usage,
    get_disk_usage,
    get_memory_usage,
    get_system_info,
    list_processes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mem(
    total: int = 16 * 1024**3,
    available: int = 8 * 1024**3,
    used: int = 8 * 1024**3,
    percent: float = 50.0,
) -> MagicMock:
    m = MagicMock()
    m.total = total
    m.available = available
    m.used = used
    m.percent = percent
    return m


def _swap(total: int = 4 * 1024**3, used: int = 1 * 1024**3, percent: float = 25.0) -> MagicMock:
    s = MagicMock()
    s.total = total
    s.used = used
    s.percent = percent
    return s


# ---------------------------------------------------------------------------
# get_system_info
# ---------------------------------------------------------------------------


class TestGetSystemInfo:
    def test_returns_expected_keys(self) -> None:
        mock_mem = _mem()
        mock_mem.total = 16 * 1024**3

        with (
            patch("rex.tools.windows_diagnostics.psutil") as mock_psutil,
            patch("rex.tools.windows_diagnostics.platform") as mock_platform,
            patch("rex.tools.windows_diagnostics.socket") as mock_socket,
        ):
            mock_psutil.virtual_memory.return_value = mock_mem
            mock_psutil.cpu_count.return_value = 8
            mock_psutil.boot_time.return_value = 1700000000.0
            mock_platform.system.return_value = "Windows"
            mock_platform.version.return_value = "10.0.19041"
            mock_platform.machine.return_value = "AMD64"
            mock_socket.gethostname.return_value = "TESTPC"

            result = get_system_info()

        assert result["platform"] == "Windows"
        assert result["platform_version"] == "10.0.19041"
        assert result["architecture"] == "AMD64"
        assert result["hostname"] == "TESTPC"
        assert result["cpu_count"] == 8
        assert isinstance(result["total_memory_gb"], float)
        assert "boot_time" in result

    def test_error_returns_error_dict(self) -> None:
        with patch("rex.tools.windows_diagnostics.psutil") as mock_psutil:
            mock_psutil.virtual_memory.side_effect = RuntimeError("no psutil")
            result = get_system_info()
        assert "error" in result


# ---------------------------------------------------------------------------
# get_cpu_usage
# ---------------------------------------------------------------------------


class TestGetCpuUsage:
    def test_returns_expected_keys(self) -> None:
        mock_freq = MagicMock()
        mock_freq.current = 3200.0

        with patch("rex.tools.windows_diagnostics.psutil") as mock_psutil:
            mock_psutil.cpu_percent.side_effect = [45.0, [40.0, 50.0]]
            mock_psutil.cpu_freq.return_value = mock_freq
            mock_psutil.cpu_count.side_effect = [8, 4]

            result = get_cpu_usage()

        assert result["usage_percent"] == 45.0
        assert result["per_core_percent"] == [40.0, 50.0]
        assert result["frequency_mhz"] == 3200.0
        assert result["cpu_count_logical"] == 8
        assert result["cpu_count_physical"] == 4

    def test_no_freq_sensor(self) -> None:
        with patch("rex.tools.windows_diagnostics.psutil") as mock_psutil:
            mock_psutil.cpu_percent.side_effect = [20.0, [20.0]]
            mock_psutil.cpu_freq.return_value = None
            mock_psutil.cpu_count.side_effect = [4, 2]

            result = get_cpu_usage()

        assert result["frequency_mhz"] is None

    def test_error_returns_error_dict(self) -> None:
        with patch("rex.tools.windows_diagnostics.psutil") as mock_psutil:
            mock_psutil.cpu_percent.side_effect = OSError("cpu error")
            result = get_cpu_usage()
        assert "error" in result


# ---------------------------------------------------------------------------
# get_memory_usage
# ---------------------------------------------------------------------------


class TestGetMemoryUsage:
    def test_returns_expected_keys(self) -> None:
        with patch("rex.tools.windows_diagnostics.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = _mem()
            mock_psutil.swap_memory.return_value = _swap()

            result = get_memory_usage()

        assert result["total_gb"] == pytest.approx(16.0, abs=0.1)
        assert result["available_gb"] == pytest.approx(8.0, abs=0.1)
        assert result["used_gb"] == pytest.approx(8.0, abs=0.1)
        assert result["percent"] == 50.0
        assert result["swap_total_gb"] == pytest.approx(4.0, abs=0.1)
        assert result["swap_used_gb"] == pytest.approx(1.0, abs=0.1)
        assert result["swap_percent"] == 25.0

    def test_error_returns_error_dict(self) -> None:
        with patch("rex.tools.windows_diagnostics.psutil") as mock_psutil:
            mock_psutil.virtual_memory.side_effect = RuntimeError("mem error")
            result = get_memory_usage()
        assert "error" in result


# ---------------------------------------------------------------------------
# get_disk_usage
# ---------------------------------------------------------------------------


class TestGetDiskUsage:
    def _make_partition(
        self, device: str = "C:\\", mountpoint: str = "C:\\", fstype: str = "NTFS"
    ) -> MagicMock:
        p = MagicMock()
        p.device = device
        p.mountpoint = mountpoint
        p.fstype = fstype
        return p

    def _make_usage(self, total: int, used: int, free: int, percent: float) -> MagicMock:
        u = MagicMock()
        u.total = total
        u.used = used
        u.free = free
        u.percent = percent
        return u

    def test_returns_partitions_list(self) -> None:
        part = self._make_partition()
        usage = self._make_usage(500 * 1024**3, 200 * 1024**3, 300 * 1024**3, 40.0)

        with patch("rex.tools.windows_diagnostics.psutil") as mock_psutil:
            mock_psutil.disk_partitions.return_value = [part]
            mock_psutil.disk_usage.return_value = usage

            result = get_disk_usage()

        assert "partitions" in result
        assert len(result["partitions"]) == 1
        p = result["partitions"][0]
        assert p["device"] == "C:\\"
        assert p["fstype"] == "NTFS"
        assert p["percent"] == 40.0

    def test_skips_inaccessible_partitions(self) -> None:
        part = self._make_partition()

        with patch("rex.tools.windows_diagnostics.psutil") as mock_psutil:
            mock_psutil.disk_partitions.return_value = [part]
            mock_psutil.disk_usage.side_effect = PermissionError("access denied")

            result = get_disk_usage()

        assert result["partitions"] == []

    def test_error_returns_error_dict(self) -> None:
        with patch("rex.tools.windows_diagnostics.psutil") as mock_psutil:
            mock_psutil.disk_partitions.side_effect = RuntimeError("disk error")
            result = get_disk_usage()
        assert "error" in result


# ---------------------------------------------------------------------------
# get_battery_status
# ---------------------------------------------------------------------------


class TestGetBatteryStatus:
    def test_battery_present(self) -> None:
        battery = MagicMock()
        battery.percent = 75.0
        battery.power_plugged = False
        battery.secsleft = 3600

        with patch("rex.tools.windows_diagnostics.psutil") as mock_psutil:
            mock_psutil.sensors_battery.return_value = battery
            result = get_battery_status()

        assert result["percent"] == 75.0
        assert result["plugged_in"] is False
        assert result["time_left_seconds"] == 3600

    def test_no_battery_sensor(self) -> None:
        with patch("rex.tools.windows_diagnostics.psutil") as mock_psutil:
            mock_psutil.sensors_battery.return_value = None
            result = get_battery_status()

        assert result["status"] == "platform_not_supported"

    def test_negative_secsleft_returns_none(self) -> None:
        battery = MagicMock()
        battery.percent = 100.0
        battery.power_plugged = True
        battery.secsleft = -1  # psutil sentinel: "unknown"

        with patch("rex.tools.windows_diagnostics.psutil") as mock_psutil:
            mock_psutil.sensors_battery.return_value = battery
            result = get_battery_status()

        assert result["time_left_seconds"] is None

    def test_attribute_error_returns_not_supported(self) -> None:
        with patch("rex.tools.windows_diagnostics.psutil") as mock_psutil:
            mock_psutil.sensors_battery.side_effect = AttributeError("no sensors_battery")
            result = get_battery_status()
        assert result["status"] == "platform_not_supported"


# ---------------------------------------------------------------------------
# list_processes
# ---------------------------------------------------------------------------


class TestListProcesses:
    def _make_proc(self, pid: int, name: str, cpu: float, mem_rss: int, status: str) -> MagicMock:
        proc = MagicMock()
        mem_info = MagicMock()
        mem_info.rss = mem_rss
        proc.info = {
            "pid": pid,
            "name": name,
            "cpu_percent": cpu,
            "memory_info": mem_info,
            "status": status,
        }
        return proc

    def test_returns_processes_sorted_by_cpu(self) -> None:
        procs = [
            self._make_proc(1, "low.exe", 1.0, 10 * 1024 * 1024, "running"),
            self._make_proc(2, "high.exe", 80.0, 200 * 1024 * 1024, "running"),
        ]

        with patch("rex.tools.windows_diagnostics.psutil") as mock_psutil:
            mock_psutil.process_iter.return_value = procs
            mock_psutil.NoSuchProcess = ProcessLookupError
            mock_psutil.AccessDenied = PermissionError

            result = list_processes()

        assert result["processes"][0]["name"] == "high.exe"
        assert result["processes"][1]["name"] == "low.exe"

    def test_skips_inaccessible_processes(self) -> None:
        bad_proc = MagicMock()
        # Make .info property raise ProcessLookupError (simulates NoSuchProcess)
        type(bad_proc).info = PropertyMock(side_effect=ProcessLookupError())

        with patch("rex.tools.windows_diagnostics.psutil") as mock_psutil:
            mock_psutil.process_iter.return_value = [bad_proc]
            mock_psutil.NoSuchProcess = ProcessLookupError
            mock_psutil.AccessDenied = PermissionError

            result = list_processes()

        # bad proc raised on .info access so it was skipped — result has 0 procs
        assert result["processes"] == []

    def test_capped_at_50(self) -> None:
        procs = [
            self._make_proc(i, f"proc{i}.exe", float(i), 1024 * 1024, "running") for i in range(100)
        ]

        with patch("rex.tools.windows_diagnostics.psutil") as mock_psutil:
            mock_psutil.process_iter.return_value = procs
            mock_psutil.NoSuchProcess = ProcessLookupError
            mock_psutil.AccessDenied = PermissionError

            result = list_processes()

        assert len(result["processes"]) == 50

    def test_error_returns_error_dict(self) -> None:
        with patch("rex.tools.windows_diagnostics.psutil") as mock_psutil:
            mock_psutil.process_iter.side_effect = RuntimeError("proc error")
            mock_psutil.NoSuchProcess = ProcessLookupError
            mock_psutil.AccessDenied = PermissionError
            result = list_processes()
        assert "error" in result


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_diagnostics_tools_registered() -> None:
    """All 6 diagnostics tools are present in the default registry."""
    from rex.tools.registry import _build_default_registry

    registry = _build_default_registry()
    names = {t.name for t in registry.all_tools()}
    for tool_name in [
        "get_system_info",
        "get_cpu_usage",
        "get_memory_usage",
        "get_disk_usage",
        "get_battery_status",
        "list_processes",
    ]:
        assert tool_name in names, f"{tool_name!r} not found in registry"


def test_diagnostics_tools_have_correct_tags() -> None:
    """All 6 diagnostics tools carry 'windows' and 'diagnostics' capability tags."""
    from rex.tools.registry import _build_default_registry

    registry = _build_default_registry()
    diag_tools = [t for t in registry.all_tools() if "diagnostics" in t.capability_tags]
    assert len(diag_tools) == 6
    for tool in diag_tools:
        assert "windows" in tool.capability_tags
        assert "diagnostics" in tool.capability_tags
