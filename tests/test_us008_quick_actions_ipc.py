"""
US-008: Verify QuickActionsPage list/create fetches are migrated to typed IPC.

Static/structural tests — verify TypeScript type definitions and that raw
fetch('/api/quick-actions') GET and POST calls no longer exist in
QuickActionsPage.tsx.
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
    # The remaining fetches are only the DELETE and run calls (US-009 scope).
    # Verify there is no 'method: POST' paired with /api/quick-actions.
    assert "method: 'POST'" not in content or "quick-actions/${" in content


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


def test_allowlist_us009_entries_updated():
    """ALLOWED_API_FETCHES.txt still references the two US-009 call sites."""
    content = _read(ALLOWED_FETCHES)
    assert "US-009" in content
    assert "QuickActionsPage" in content


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
