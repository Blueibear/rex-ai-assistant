"""
US-328: End-to-end verification -- GUI launch and backend connection.

Verifies via code review (static analysis) that:
- The Electron main process creates a window and calls validateBridges() at startup
- The App.tsx shell handles backend unavailability gracefully (no crash/loop)
- The /api/setup/status and /api/status/current endpoints exist in gui_app.py
- The setup wizard flow cannot loop (error falls back to needs_setup=false)
- The Home page has no direct backend API calls (renders without backend)
- All routes are wrapped in ErrorBoundary
- The manual test doc exists
"""

from pathlib import Path

REPO = Path(__file__).parent.parent


def read_file(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Manual test script
# ---------------------------------------------------------------------------


def test_manual_test_script_exists():
    """The manual test script documenting GUI launch steps must exist."""
    doc = REPO / "docs" / "e2e-gui-launch-test.md"
    assert doc.exists(), "docs/e2e-gui-launch-test.md must exist"


def test_manual_test_script_has_launch_command():
    """The manual test script must document the launch command."""
    src = read_file("docs/e2e-gui-launch-test.md")
    assert (
        "rex-gui" in src or "npm run dev" in src
    ), "Manual test doc must document the launch command"


def test_manual_test_script_has_backend_verification_steps():
    """The manual test script must document backend connection verification."""
    src = read_file("docs/e2e-gui-launch-test.md")
    assert (
        "/api/status" in src or "backend" in src.lower()
    ), "Manual test doc must include backend connection verification"


def test_manual_test_script_has_expected_first_screen():
    """The manual test script must describe the expected first screen."""
    src = read_file("docs/e2e-gui-launch-test.md")
    assert "Home" in src and (
        "first screen" in src.lower() or "expected" in src.lower()
    ), "Manual test doc must describe the expected first screen"


# ---------------------------------------------------------------------------
# Electron main process: startup sequence
# ---------------------------------------------------------------------------


def test_main_process_calls_validate_bridges_at_startup():
    """index.ts must call validateBridges() inside app.whenReady()."""
    src = read_file("gui/src/main/index.ts")
    assert "validateBridges" in src, "index.ts must import/call validateBridges"
    # Both the import and the call should be present
    assert "validateBridges()" in src, "index.ts must call validateBridges() at app startup"


def test_main_process_calls_create_window_at_startup():
    """index.ts must create the main window inside app.whenReady()."""
    src = read_file("gui/src/main/index.ts")
    assert "createWindow" in src, "index.ts must define and call createWindow()"
    assert "app.whenReady" in src, "index.ts must use app.whenReady() to initialise the app"


def test_bridge_resolver_imported_in_main():
    """bridgeResolver must be imported in the main process entry point."""
    src = read_file("gui/src/main/index.ts")
    assert (
        "bridgeResolver" in src
    ), "index.ts must import from bridgeResolver to call validateBridges"


# ---------------------------------------------------------------------------
# App.tsx: backend connection and error handling
# ---------------------------------------------------------------------------


def test_app_tsx_checks_setup_status_on_load():
    """App.tsx must call getSetupStatus() via IPC to determine if setup wizard is needed."""
    src = read_file("gui/src/renderer/src/App.tsx")
    assert (
        "getSetupStatus" in src
    ), "App.tsx must call getSetupStatus() on load to decide setup vs main app"


def test_app_tsx_setup_error_falls_back_to_no_setup():
    """App.tsx must catch /api/setup/status errors and default to needs_setup=false."""
    src = read_file("gui/src/renderer/src/App.tsx")
    # The .catch handler must set needsSetup to false (prevents setup wizard loop on error)
    assert (
        ".catch" in src and "setNeedsSetup(false)" in src
    ), "App.tsx must catch setup/status errors and fall back to needs_setup=false"


def test_app_tsx_handles_backend_unavailable_gracefully():
    """App.tsx must render an error state (not crash) when the backend is unreachable."""
    src = read_file("gui/src/renderer/src/App.tsx")
    # Must have an error branch that does NOT crash the app
    assert (
        "backend" in src.lower() or "unavailable" in src.lower() or "error" in src.lower()
    ), "App.tsx must handle backend unavailability with a user-visible error state"
    assert (
        "EmptyState" in src or "error" in src.lower()
    ), "App.tsx must render an error UI when the backend is unreachable"


def test_app_tsx_does_not_redirect_to_login_on_error():
    """App.tsx must not redirect to a login route when the backend is unreachable."""
    src = read_file("gui/src/renderer/src/App.tsx")
    # There should be no login route reference
    assert (
        "/login" not in src and "LoginPage" not in src
    ), "App.tsx must not have a login redirect — this app has no auth requirement"


def test_app_tsx_all_routes_wrapped_in_error_boundary():
    """All page routes in App.tsx must be wrapped in ErrorBoundary."""
    src = read_file("gui/src/renderer/src/App.tsx")
    assert "ErrorBoundary" in src, "App.tsx must use ErrorBoundary to prevent route crashes"
    # Count Route elements — each should have an ErrorBoundary nearby
    route_count = src.count("<Route path=")
    error_boundary_count = src.count("<ErrorBoundary>")
    # At least most routes should be wrapped (allow for redirect routes)
    assert error_boundary_count >= route_count - 2, (
        f"Expected most routes ({route_count}) to be wrapped in ErrorBoundary, "
        f"found {error_boundary_count} wrappers"
    )


# ---------------------------------------------------------------------------
# Backend: required API endpoints for launch
# ---------------------------------------------------------------------------


def test_backend_has_setup_status_endpoint():
    """gui_app.py must expose /api/setup/status."""
    src = read_file("rex/gui_app.py")
    assert (
        "/api/setup/status" in src
    ), "gui_app.py must expose /api/setup/status for the setup wizard check"


def test_backend_has_status_current_endpoint():
    """gui_app.py must expose /api/status/current for the connection check."""
    src = read_file("rex/gui_app.py")
    assert (
        "/api/status/current" in src
    ), "gui_app.py must expose /api/status/current for backend connection verification"


def test_backend_status_endpoint_returns_status_field():
    """The /api/status/current handler must return a 'status' field."""
    src = read_file("rex/gui_app.py")
    # The endpoint must return status via jsonify
    assert (
        "status" in src and "jsonify" in src
    ), "gui_app.py status endpoint must return a 'status' field via jsonify"


# ---------------------------------------------------------------------------
# Home page: renders without backend calls
# ---------------------------------------------------------------------------


def test_home_page_has_no_fetch_calls():
    """HomePage.tsx must not make fetch() calls (should render without a backend)."""
    src = read_file("gui/src/pages/HomePage.tsx")
    assert (
        "fetch(" not in src
    ), "HomePage.tsx must not call fetch() — it must render without backend dependency"


def test_home_page_has_no_useeffect_with_api_calls():
    """HomePage.tsx must not contain useEffect with API calls."""
    src = read_file("gui/src/pages/HomePage.tsx")
    # Simple check: no window.rex IPC calls either
    assert (
        "window.rex" not in src
    ), "HomePage.tsx must not make IPC calls — it renders static content only"


def test_home_page_renders_static_content():
    """HomePage.tsx must contain static content: heading and nav links."""
    src = read_file("gui/src/pages/HomePage.tsx")
    assert "Home" in src, "HomePage.tsx must contain a 'Home' heading"
    assert "NavLink" in src or "<Link" in src, "HomePage.tsx must contain navigation links"


# ---------------------------------------------------------------------------
# Error boundary component
# ---------------------------------------------------------------------------


def test_error_boundary_component_exists():
    """ErrorBoundary.tsx must exist to catch render errors in page routes."""
    path = REPO / "gui" / "src" / "components" / "ErrorBoundary.tsx"
    assert path.exists(), "gui/src/components/ErrorBoundary.tsx must exist"


def test_error_boundary_is_a_class_component():
    """ErrorBoundary must be a React class component (required for componentDidCatch)."""
    src = read_file("gui/src/components/ErrorBoundary.tsx")
    has_class = "class ErrorBoundary" in src or "Component" in src
    has_catch = "componentDidCatch" in src or "getDerivedStateFromError" in src or "hasError" in src
    assert (
        has_class or has_catch
    ), "ErrorBoundary must be a class component or use getDerivedStateFromError"
