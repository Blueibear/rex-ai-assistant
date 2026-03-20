"""Windows service wrapper for Rex lean nodes (pywin32 required)."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING, Any

if sys.platform != "win32":
    raise ImportError(
        "rex.windows_service is only supported on Windows. "
        "pywin32 is required: pip install pywin32"
    )

try:
    import servicemanager  # noqa: E402
    import win32event  # noqa: E402
    import win32service  # noqa: E402
    import win32serviceutil  # noqa: E402

    _PYWIN32_SERVICE_AVAILABLE = True
except ImportError:  # pragma: no cover - pywin32 service components optional
    servicemanager = None
    win32event = None
    win32service = None
    win32serviceutil = None
    _PYWIN32_SERVICE_AVAILABLE = False

DEFAULT_SERVICES = "event_bus,workflow_runner,memory_store,credential_manager"
DEFAULT_PORT = "8765"

_ServiceBase: Any
if TYPE_CHECKING:
    import win32serviceutil as _win32serviceutil_t

    _ServiceBase = _win32serviceutil_t.ServiceFramework
else:
    _ServiceBase = win32serviceutil.ServiceFramework if _PYWIN32_SERVICE_AVAILABLE else object


class RexNodeService(_ServiceBase):
    _svc_name_ = "RexLeanNode"
    _svc_display_name_ = "Rex Lean Node"
    _svc_description_ = "Rex AI Assistant lean node service."

    def __init__(self, args):
        super().__init__(args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.process: subprocess.Popen[bytes] | None = None

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.stop_event)
        if self.process:
            self.process.terminate()

    def SvcDoRun(self):
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, ""),
        )
        services = os.environ.get("REX_SERVICES", DEFAULT_SERVICES)
        port = os.environ.get("REX_SERVICE_PORT", DEFAULT_PORT)
        cmd = [
            sys.executable,
            "-m",
            "rex.app",
            "--services",
            services,
            "--port",
            port,
        ]
        self.process = subprocess.Popen(cmd)
        win32event.WaitForSingleObject(self.stop_event, win32event.INFINITE)


def main() -> None:
    if not _PYWIN32_SERVICE_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "pywin32 service components are required to run Rex as a Windows service. "
            "Install via: pip install pywin32"
        )
    win32serviceutil.HandleCommandLine(RexNodeService)


if __name__ == "__main__":
    main()
