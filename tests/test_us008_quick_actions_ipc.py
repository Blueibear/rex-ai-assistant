"""
US-008 / US-009: Verify QuickActionsPage IPC migration is complete.

Static/structural tests — verify TypeScript type definitions and that raw
fetch('/api/quick-actions') calls no longer exist in QuickActionsPage.tsx.
US-008 migrated list/create; US-009 migrated delete/run.
"""

from pathlib import Path

IPC_TYPES = Path("gui/src/types/ipc.ts")
HANDLER_FILE = Path("gui/src/main/handlers/quickActions.ts")
PRELOAD_FILE = Path("gui/src/preload/index.ts")
IPC_AGGREGATOR = Path("gui/src/main/ipc.ts")
QUICK_ACTIONS_PAGE = Path("gui/src/pages/QuickActionsPage.tsx")
ALLOWED_FETCHES = Path("gui/src/ALLOWED_API_FETCHES.txt")
BRIDGE_SCRIPT = Path("bridge/rex_quick_actions_bridge.py")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_quick_action_interface_defined_in_ipc_types():
    """QuickAction interface is defined in ipc.ts."""
    content = _read(IPC_TYPES)
    assert "QuickAction" in content
    assert "label: string" in content
    assert "command: string" in content


def test_list_quick_actions_in_rex_api():
    """RexAPI interface declares listQuickActions."""
    content = _read(IPC_TYPES)
    assert "listQuickActions" in content


def test_create_quick_action_in_rex_api():
    """RexAPI interface declares createQuickAction."""
    content = _read(IPC_TYPES)
    assert "createQuickAction" in content


def test_handler_file_exists():
    """quickActions.ts handler file exists."""
    assert HANDLER_FILE.exists()


def test_handler_registers_list_channel():
    """Main-process handler registers rex:listQuickActions."""
    content = _read(HANDLER_FILE)
    assert "rex:listQuickActions" in content


def test_handler_registers_create_channel():
    """Main-process handler registers rex:createQuickAction."""
    content = _read(HANDLER_FILE)
    assert "rex:createQuickAction" in content


def test_handler_calls_bridge():
    """Main-process handler invokes the quick actions bridge script."""
    content = _read(HANDLER_FILE)
    assert "rex_quick_actions_bridge.py" in content


def test_ipc_aggregator_imports_handler():
    """ipc.ts imports registerQuickActionsHandlers."""
    content = _read(IPC_AGGREGATOR)
    assert "registerQuickActionsHandlers" in content


def test_ipc_aggregator_calls_handler():
    """ipc.ts calls registerQuickActionsHandlers()."""
    content = _read(IPC_AGGREGATOR)
    assert "registerQuickActionsHandlers()" in content


def test_preload_exposes_list_quick_actions():
    """Preload exposes listQuickActions to the renderer."""
    content = _read(PRELOAD_FILE)
    assert "listQuickActions" in content
    assert "rex:listQuickActions" in content


def test_preload_exposes_create_quick_action():
    """Preload exposes createQuickAction to the renderer."""
    content = _read(PRELOAD_FILE)
    assert "createQuickAction" in content
    assert "rex:createQuickAction" in content


def test_quick_actions_page_no_raw_get_fetch():
    """QuickActionsPage.tsx has no raw fetch('/api/quick-actions') GET call."""
    content = _read(QUICK_ACTIONS_PAGE)
    assert "fetch('/api/quick-actions'" not in content
    assert 'fetch("/api/quick-actions"' not in content


def test_quick_actions_page_no_raw_post_fetch():
    """QuickActionsPage.tsx has no raw POST fetch to /api/quick-actions."""
    content = _read(QUICK_ACTIONS_PAGE)
    assert "fetch(" not in content or "/api/quick-actions" not in content


def test_quick_actions_page_uses_list_ipc():
    """QuickActionsPage.tsx calls window.rex.listQuickActions()."""
    content = _read(QUICK_ACTIONS_PAGE)
    assert "window.rex.listQuickActions" in content


def test_quick_actions_page_uses_create_ipc():
    """QuickActionsPage.tsx calls window.rex.createQuickAction(...)."""
    content = _read(QUICK_ACTIONS_PAGE)
    assert "window.rex.createQuickAction" in content


def test_allowlist_no_us008_entries():
    """ALLOWED_API_FETCHES.txt no longer has the US-008 list/add entries."""
    content = _read(ALLOWED_FETCHES)
    assert "US-008" not in content


def test_allowlist_us009_entries_removed():
    """ALLOWED_API_FETCHES.txt no longer has US-009 QuickActionsPage entries (US-009 migrated)."""
    content = _read(ALLOWED_FETCHES)
    assert "US-009" not in content
    assert "QuickActionsPage" not in content


def test_bridge_script_exists():
    """bridge/rex_quick_actions_bridge.py exists."""
    assert BRIDGE_SCRIPT.exists()


def test_bridge_script_supports_list_command():
    """Bridge script handles the 'list' command."""
    content = _read(BRIDGE_SCRIPT)
    assert '"list"' in content or "== 'list'" in content or '== "list"' in content


def test_bridge_script_supports_add_command():
    """Bridge script handles the 'add' command."""
    content = _read(BRIDGE_SCRIPT)
    assert '"add"' in content or "== 'add'" in content or '== "add"' in content


# US-009 additions — delete and run migration
def test_delete_quick_action_in_rex_api():
    """RexAPI interface declares deleteQuickAction."""
    content = _read(IPC_TYPES)
    assert "deleteQuickAction" in content


def test_run_quick_action_in_rex_api():
    """RexAPI interface declares runQuickAction with discriminated status."""
    content = _read(IPC_TYPES)
    assert "runQuickAction" in content
    assert "QuickActionRunResponse" in content
    assert "'attempted'" in content or '"attempted"' in content


def test_handler_registers_delete_channel():
    """Main-process handler registers rex:deleteQuickAction."""
    content = _read(HANDLER_FILE)
    assert "rex:deleteQuickAction" in content


def test_handler_registers_run_channel():
    """Main-process handler registers rex:runQuickAction."""
    content = _read(HANDLER_FILE)
    assert "rex:runQuickAction" in content


def test_preload_exposes_delete_quick_action():
    """Preload exposes deleteQuickAction to the renderer."""
    content = _read(PRELOAD_FILE)
    assert "deleteQuickAction" in content
    assert "rex:deleteQuickAction" in content


def test_preload_exposes_run_quick_action():
    """Preload exposes runQuickAction to the renderer."""
    content = _read(PRELOAD_FILE)
    assert "runQuickAction" in content
    assert "rex:runQuickAction" in content


def test_quick_actions_page_uses_delete_ipc():
    """QuickActionsPage.tsx calls window.rex.deleteQuickAction(...)."""
    content = _read(QUICK_ACTIONS_PAGE)
    assert "window.rex.deleteQuickAction" in content


def test_quick_actions_page_uses_run_ipc():
    """QuickActionsPage.tsx calls window.rex.runQuickAction(...)."""
    content = _read(QUICK_ACTIONS_PAGE)
    assert "window.rex.runQuickAction" in content


def test_quick_actions_page_no_auth_headers():
    """QuickActionsPage.tsx no longer defines authHeaders (all calls use IPC)."""
    content = _read(QUICK_ACTIONS_PAGE)
    assert "authHeaders" not in content


def test_bridge_script_supports_delete_command():
    """Bridge script handles the 'delete' command."""
    content = _read(BRIDGE_SCRIPT)
    assert '"delete"' in content or "== 'delete'" in content or '== "delete"' in content


def test_bridge_script_supports_run_command():
    """Bridge script handles the 'run' command."""
    content = _read(BRIDGE_SCRIPT)
    assert '"run"' in content or "== 'run'" in content or '== "run"' in content
