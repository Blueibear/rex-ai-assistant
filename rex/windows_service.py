"""Windows service wrapper for Rex lean nodes (pywin32 required)."""

from __future__ import annotations

import os
import subprocess
import sys

if sys.platform != "win32":
    raise ImportError(
        "rex.windows_service is only supported on Windows. "
        "pywin32 is required: pip install pywin32"
    )

import servicemanager  # noqa: E402
import win32event  # noqa: E402
import win32service  # noqa: E402
import win32serviceutil  # noqa: E402

DEFAULT_SERVICES = "event_bus,workflow_runner,memory_store,credential_manager"
DEFAULT_PORT = "8765"


class RexNodeService(win32serviceutil.ServiceFramework):
    _svc_name_ = "RexLeanNode"
    _svc_display_name_ = "Rex Lean Node"
    _svc_description_ = "Rex AI Assistant lean node service."

    def __init__(self, args):
        super().__init__(args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.process: subprocess.Popen[str] | None = None

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
    win32serviceutil.HandleCommandLine(RexNodeService)


if __name__ == "__main__":
    main()
