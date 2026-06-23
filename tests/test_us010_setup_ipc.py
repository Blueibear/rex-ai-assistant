"""
US-010: Verify SetupWizardPage and App.tsx /api/setup/* IPC migration is complete.

Static/structural tests — verify TypeScript type definitions and that raw
fetch('/api/setup/') calls no longer exist.
"""

from pathlib import Path

IPC_TYPES = Path("gui/src/types/ipc.ts")
HANDLER_FILE = Path("gui/src/main/handlers/setup.ts")
IPC_AGGREGATOR = Path("gui/src/main/ipc.ts")
PRELOAD_FILE = Path("gui/src/preload/index.ts")
APP_TSX = Path("gui/src/renderer/src/App.tsx")
SETUP_WIZARD_PAGE = Path("gui/src/pages/SetupWizardPage.tsx")
ALLOWED_FETCHES = Path("gui/src/ALLOWED_API_FETCHES.txt")
BRIDGE_SCRIPT = Path("bridge/rex_setup_bridge.py")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_setup_status_response_in_ipc_types():
    """SetupStatusResponse interface is defined in ipc.ts."""
    content = _read(IPC_TYPES)
    assert "SetupStatusResponse" in content
    assert "needs_setup" in content


def test_setup_complete_payload_in_ipc_types():
    """SetupCompletePayload interface is defined in ipc.ts."""
    content = _read(IPC_TYPES)
    assert "SetupCompletePayload" in content


def test_setup_complete_response_in_ipc_types():
    """SetupCompleteResponse interface is defined in ipc.ts."""
    content = _read(IPC_TYPES)
    assert "SetupCompleteResponse" in content


def test_get_setup_status_in_rex_api():
    """RexAPI declares getSetupStatus."""
    content = _read(IPC_TYPES)
    assert "getSetupStatus" in content


def test_complete_setup_in_rex_api():
    """RexAPI declares completeSetup."""
    content = _read(IPC_TYPES)
    assert "completeSetup" in content


def test_handler_file_exists():
    """gui/src/main/handlers/setup.ts exists."""
    assert HANDLER_FILE.exists()


def test_handler_registers_get_setup_status_channel():
    """Main-process handler registers rex:getSetupStatus."""
    content = _read(HANDLER_FILE)
    assert "rex:getSetupStatus" in content


def test_handler_registers_complete_setup_channel():
    """Main-process handler registers rex:completeSetup."""
    content = _read(HANDLER_FILE)
    assert "rex:completeSetup" in content


def test_handler_calls_bridge():
    """Main-process handler invokes the setup bridge script."""
    content = _read(HANDLER_FILE)
    assert "rex_setup_bridge.py" in content


def test_ipc_aggregator_imports_handler():
    """ipc.ts imports registerSetupHandlers."""
    content = _read(IPC_AGGREGATOR)
    assert "registerSetupHandlers" in content


def test_ipc_aggregator_calls_handler():
    """ipc.ts calls registerSetupHandlers()."""
    content = _read(IPC_AGGREGATOR)
    assert "registerSetupHandlers()" in content


def test_preload_exposes_get_setup_status():
    """Preload exposes getSetupStatus to the renderer."""
    content = _read(PRELOAD_FILE)
    assert "getSetupStatus" in content
    assert "rex:getSetupStatus" in content


def test_preload_exposes_complete_setup():
    """Preload exposes completeSetup to the renderer."""
    content = _read(PRELOAD_FILE)
    assert "completeSetup" in content
    assert "rex:completeSetup" in content


def test_app_tsx_no_raw_setup_status_fetch():
    """App.tsx has no raw fetch('/api/setup/status') call."""
    content = _read(APP_TSX)
    assert "/api/setup/status" not in content


def test_app_tsx_uses_get_setup_status_ipc():
    """App.tsx calls getSetupStatus() via window.rex."""
    content = _read(APP_TSX)
    assert "getSetupStatus" in content


def test_setup_wizard_page_no_raw_complete_fetch():
    """SetupWizardPage.tsx has no raw fetch('/api/setup/complete') call."""
    content = _read(SETUP_WIZARD_PAGE)
    assert "/api/setup/complete" not in content
    assert "fetch(" not in content or "/api/setup" not in content


def test_setup_wizard_page_uses_complete_setup_ipc():
    """SetupWizardPage.tsx calls window.rex.completeSetup(...)."""
    content = _read(SETUP_WIZARD_PAGE)
    assert "window.rex.completeSetup" in content


def test_allowlist_no_setup_wizard_entries():
    """ALLOWED_API_FETCHES.txt no longer has SetupWizardPage.tsx entries."""
    content = _read(ALLOWED_FETCHES)
    assert "SetupWizardPage" not in content


def test_allowlist_no_app_tsx_entries():
    """ALLOWED_API_FETCHES.txt no longer has App.tsx entries."""
    content = _read(ALLOWED_FETCHES)
    assert "App.tsx" not in content


def test_allowlist_no_us010_entries():
    """ALLOWED_API_FETCHES.txt no longer mentions US-010."""
    content = _read(ALLOWED_FETCHES)
    assert "US-010" not in content


def test_bridge_script_exists():
    """bridge/rex_setup_bridge.py exists."""
    assert BRIDGE_SCRIPT.exists()


def test_bridge_script_supports_status_command():
    """Bridge script handles the 'status' command."""
    content = _read(BRIDGE_SCRIPT)
    assert '"status"' in content or "== 'status'" in content or '== "status"' in content


def test_bridge_script_supports_complete_command():
    """Bridge script handles the 'complete' command."""
    content = _read(BRIDGE_SCRIPT)
    assert '"complete"' in content or "== 'complete'" in content or '== "complete"' in content


def test_bridge_script_calls_create_user():
    """Bridge script calls rex.auth.create_user."""
    content = _read(BRIDGE_SCRIPT)
    assert "create_user" in content


def test_bridge_script_calls_bootstrap_admin():
    """Bridge script calls bootstrap_admin_if_first_user."""
    content = _read(BRIDGE_SCRIPT)
    assert "bootstrap_admin_if_first_user" in content


def test_bridge_script_writes_env_secrets():
    """Bridge script writes env secrets via _write_env_secrets."""
    content = _read(BRIDGE_SCRIPT)
    assert "_write_env_secrets" in content
