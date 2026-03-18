"""Tests for US-149: Scaffold GUI application shell.

Acceptance criteria:
- GUI application launches with a single command (rex-gui entry point)
- main window renders with a left sidebar and a main content area
- sidebar contains placeholder navigation items: Chat, Voice, Schedule, Overview
- window is resizable and has a minimum usable size (800x600)
- application closes cleanly without errors
- Typecheck passes
- Verify changes work in browser
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
HTML_PATH = REPO_ROOT / "rex" / "dashboard" / "templates" / "index.html"
CSS_PATH = REPO_ROOT / "rex" / "dashboard" / "static" / "css" / "dashboard.css"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def _html() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


def _css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def _pyproject() -> str:
    return PYPROJECT_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


class TestEntryPoint:
    def test_rex_gui_entry_point_in_pyproject(self) -> None:
        """rex-gui entry point is declared in pyproject.toml."""
        assert 'rex-gui = "rex.gui_app:main"' in _pyproject()

    def test_gui_app_module_importable(self) -> None:
        """rex.gui_app can be imported without errors."""
        mod = importlib.import_module("rex.gui_app")
        assert mod is not None

    def test_gui_app_has_main_function(self) -> None:
        """rex.gui_app exposes a callable main()."""
        mod = importlib.import_module("rex.gui_app")
        assert callable(getattr(mod, "main", None))

    def test_gui_app_has_create_flask_app(self) -> None:
        """rex.gui_app has _create_flask_app helper."""
        mod = importlib.import_module("rex.gui_app")
        assert callable(getattr(mod, "_create_flask_app", None))

    def test_create_flask_app_returns_flask_app(self) -> None:
        """_create_flask_app() returns a Flask application object."""
        mod = importlib.import_module("rex.gui_app")
        app = mod._create_flask_app()
        assert hasattr(app, "run")  # Flask apps have run()
        assert hasattr(app, "register_blueprint")


# ---------------------------------------------------------------------------
# Sidebar navigation items
# ---------------------------------------------------------------------------


class TestSidebarNavigation:
    def test_chat_nav_link_present(self) -> None:
        """Sidebar contains a Chat navigation link."""
        html = _html()
        assert 'data-section="chat"' in html

    def test_voice_nav_link_present(self) -> None:
        """Sidebar contains a Voice navigation link."""
        html = _html()
        assert 'data-section="voice"' in html

    def test_schedule_nav_link_present(self) -> None:
        """Sidebar contains a Schedule navigation link."""
        html = _html()
        assert 'data-section="schedule"' in html

    def test_overview_nav_link_present(self) -> None:
        """Sidebar contains an Overview navigation link."""
        html = _html()
        assert 'data-section="overview"' in html

    def test_chat_nav_link_text(self) -> None:
        """Chat nav link displays 'Chat' text."""
        html = _html()
        assert re.search(r'data-section="chat"[^>]*>.*?Chat', html, re.DOTALL)

    def test_voice_nav_link_text(self) -> None:
        """Voice nav link displays 'Voice' text."""
        html = _html()
        assert re.search(r'data-section="voice"[^>]*>.*?Voice', html, re.DOTALL)

    def test_schedule_nav_link_text(self) -> None:
        """Schedule nav link displays 'Schedule' text."""
        html = _html()
        assert re.search(r'data-section="schedule"[^>]*>.*?Schedule', html, re.DOTALL)

    def test_overview_nav_link_text(self) -> None:
        """Overview nav link displays 'Overview' text."""
        html = _html()
        assert re.search(r'data-section="overview"[^>]*>.*?Overview', html, re.DOTALL)

    def test_nav_links_are_in_sidebar(self) -> None:
        """Navigation links are inside a sidebar element."""
        html = _html()
        # Find the sidebar block and confirm it has nav links
        sidebar_match = re.search(r'class="sidebar[^"]*"(.*?)</nav>', html, re.DOTALL)
        assert sidebar_match is not None, "sidebar nav element not found"
        sidebar_html = sidebar_match.group(1)
        for section in ("chat", "voice", "schedule", "overview"):
            assert f'data-section="{section}"' in sidebar_html


# ---------------------------------------------------------------------------
# Content sections exist in the HTML
# ---------------------------------------------------------------------------


class TestContentSections:
    def test_chat_section_exists(self) -> None:
        """chat-section element exists in HTML."""
        assert 'id="chat-section"' in _html()

    def test_voice_section_exists(self) -> None:
        """voice-section placeholder element exists in HTML."""
        assert 'id="voice-section"' in _html()

    def test_schedule_section_exists(self) -> None:
        """schedule-section placeholder element exists in HTML."""
        assert 'id="schedule-section"' in _html()

    def test_overview_section_exists(self) -> None:
        """overview-section placeholder element exists in HTML."""
        assert 'id="overview-section"' in _html()

    def test_main_content_area_exists(self) -> None:
        """A main content area element exists."""
        html = _html()
        assert 'class="main-content"' in html or "<main" in html

    def test_voice_section_has_heading(self) -> None:
        """Voice placeholder section has an h1 heading."""
        html = _html()
        match = re.search(r'id="voice-section".*?<h1>(.*?)</h1>', html, re.DOTALL)
        assert match is not None
        assert "Voice" in match.group(1)

    def test_schedule_section_has_heading(self) -> None:
        """Schedule placeholder section has an h1 heading."""
        html = _html()
        match = re.search(r'id="schedule-section".*?<h1>(.*?)</h1>', html, re.DOTALL)
        assert match is not None
        assert "Schedule" in match.group(1)

    def test_overview_section_has_heading(self) -> None:
        """Overview placeholder section has an h1 heading."""
        html = _html()
        match = re.search(r'id="overview-section".*?<h1>(.*?)</h1>', html, re.DOTALL)
        assert match is not None
        assert "Overview" in match.group(1)


# ---------------------------------------------------------------------------
# Minimum window size
# ---------------------------------------------------------------------------


class TestMinimumSize:
    def test_css_has_min_width_800(self) -> None:
        """CSS defines a min-width of at least 800px for the app."""
        css = _css()
        assert "min-width: 800px" in css

    def test_css_has_min_height_600(self) -> None:
        """CSS defines a min-height of at least 600px for the app."""
        css = _css()
        assert "min-height: 600px" in css

    def test_css_min_size_on_app_element(self) -> None:
        """The min-width and min-height are set on #app."""
        css = _css()
        # Find the #app block that contains both min-width and min-height
        re.search(r"#app\s*\{([^}]+)\}", css)
        # There may be multiple #app blocks; check any of them
        app_blocks = re.findall(r"#app\s*\{([^}]+)\}", css)
        combined = " ".join(app_blocks)
        assert "min-width" in combined
        assert "min-height" in combined


# ---------------------------------------------------------------------------
# Flask app registers dashboard blueprint (routes reachable)
# ---------------------------------------------------------------------------


class TestFlaskAppRoutes:
    def test_dashboard_route_registered(self) -> None:
        """Flask app has a /dashboard route registered."""
        mod = importlib.import_module("rex.gui_app")
        app = mod._create_flask_app()
        with app.test_client() as client:
            resp = client.get("/dashboard")
            # Should not 404; could be 200 or 302 redirect
            assert resp.status_code != 404

    def test_status_api_route_registered(self) -> None:
        """Flask app has /api/dashboard/status route."""
        mod = importlib.import_module("rex.gui_app")
        app = mod._create_flask_app()
        with app.test_client() as client:
            resp = client.get("/api/dashboard/status")
            assert resp.status_code in (200, 401, 403)

    def test_app_closes_cleanly_on_system_exit(self) -> None:
        """main() catches SystemExit from app.run() without raising."""
        from unittest.mock import MagicMock, patch

        mod = importlib.import_module("rex.gui_app")

        # Patch app.run to raise SystemExit, patch webbrowser/threading
        mock_app = MagicMock()
        mock_app.run.side_effect = SystemExit(0)

        with (
            patch.object(mod, "_create_flask_app", return_value=mock_app),
            patch("threading.Thread"),
            patch("webbrowser.open"),
            patch("signal.signal"),
        ):
            # Should not raise
            mod.main()
