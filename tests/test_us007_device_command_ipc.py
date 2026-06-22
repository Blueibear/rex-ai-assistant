"""
US-007: Verify sendDeviceCommand IPC type definitions are correct.

These are static/structural tests — they verify the TypeScript type definitions
without running the Electron process.  Runtime end-to-end verification (actual
HA state change) is deferred to US-048.
"""

import re
from pathlib import Path

IPC_TYPES = Path("gui/src/types/ipc.ts")
HANDLER_FILE = Path("gui/src/main/handlers/devices.ts")
HA_FILE = Path("gui/src/main/homeAssistant.ts")
PRELOAD_FILE = Path("gui/src/preload/index.ts")
DEVICES_PAGE = Path("gui/src/pages/DevicesPage.tsx")
ALLOWED_FETCHES = Path("gui/src/ALLOWED_API_FETCHES.txt")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_device_command_status_type_defined():
    """DeviceCommandStatus discriminated union is defined in ipc.ts."""
    content = _read(IPC_TYPES)
    assert "DeviceCommandStatus" in content
    assert "'attempted'" in content
    assert "'completed'" in content
    assert "'verified'" in content
    assert "'failed'" in content


def test_device_command_response_interface_defined():
    """DeviceCommandResponse interface with status field is defined in ipc.ts."""
    content = _read(IPC_TYPES)
    assert "DeviceCommandResponse" in content
    # Interface must reference the discriminated status type
    assert re.search(r"status:\s*DeviceCommandStatus", content)


def test_send_device_command_in_rex_api():
    """RexAPI interface declares sendDeviceCommand returning DeviceCommandResponse."""
    content = _read(IPC_TYPES)
    assert "sendDeviceCommand" in content
    assert "DeviceCommandResponse" in content


def test_handler_registers_ipc_channel():
    """Main-process handler registers rex:sendDeviceCommand."""
    content = _read(HANDLER_FILE)
    assert "rex:sendDeviceCommand" in content


def test_handler_calls_call_device_command():
    """Main-process handler delegates to callDeviceCommand from homeAssistant."""
    content = _read(HANDLER_FILE)
    assert "callDeviceCommand" in content


def test_ha_module_exports_call_device_command():
    """homeAssistant.ts exports callDeviceCommand."""
    content = _read(HA_FILE)
    assert "export async function callDeviceCommand" in content


def test_ha_module_exports_device_command_response():
    """homeAssistant.ts exports DeviceCommandResponse."""
    content = _read(HA_FILE)
    assert "DeviceCommandResponse" in content


def test_response_shape_includes_failed_status():
    """callDeviceCommand returns { status: 'failed' } when HA is not configured."""
    content = _read(HA_FILE)
    assert "status: 'failed'" in content


def test_response_shape_includes_attempted_status():
    """callDeviceCommand returns { status: 'attempted' } on HTTP success."""
    content = _read(HA_FILE)
    assert "status: 'attempted'" in content


def test_preload_exposes_send_device_command():
    """Preload exposes sendDeviceCommand to the renderer."""
    content = _read(PRELOAD_FILE)
    assert "sendDeviceCommand" in content
    assert "rex:sendDeviceCommand" in content


def test_devices_page_no_raw_api_fetch():
    """DevicesPage.tsx contains no raw fetch('/api/devices/.../command') call."""
    content = _read(DEVICES_PAGE)
    assert "/api/devices/" not in content
    assert "fetch(" not in content


def test_allowlist_does_not_contain_devices_command():
    """ALLOWED_API_FETCHES.txt no longer lists the DevicesPage command fetch."""
    content = _read(ALLOWED_FETCHES)
    assert "DevicesPage" not in content or "command" not in content
