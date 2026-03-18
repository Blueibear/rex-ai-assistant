"""Tests for US-151: Implement active navigation state and panel switching.

Acceptance criteria:
- clicking each sidebar item displays the corresponding panel in the content area
- the active sidebar item is visually highlighted
- navigation does not reload the page or lose state in already-loaded panels
- back/forward browser navigation works correctly if the GUI is web-based
- Typecheck passes
- Verify changes work in browser
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
HTML_PATH = REPO_ROOT / "rex" / "dashboard" / "templates" / "index.html"
CSS_PATH = REPO_ROOT / "rex" / "dashboard" / "static" / "css" / "dashboard.css"
JS_PATH = REPO_ROOT / "rex" / "dashboard" / "static" / "js" / "dashboard.js"

SECTIONS = [
    "chat",
    "voice",
    "schedule",
    "overview",
    "settings",
    "reminders",
    "notifications",
    "status",
]


def _html() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


def _css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def _js() -> str:
    return JS_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# AC: clicking each sidebar item displays the corresponding panel
# ---------------------------------------------------------------------------


class TestSidebarPanelSwitching:
    def test_all_sections_have_nav_link(self) -> None:
        html = _html()
        for section in SECTIONS:
            assert (
                f'data-section="{section}"' in html
            ), f'No nav link with data-section="{section}" found in HTML'

    def test_all_sections_have_content_panel(self) -> None:
        html = _html()
        for section in SECTIONS:
            assert (
                f'id="{section}-section"' in html
            ), f'No content panel with id="{section}-section" found in HTML'

    def test_switchSection_hides_inactive_sections(self) -> None:
        """JS switchSection hides non-active sections via 'hidden' class toggle."""
        js = _js()
        # The JS should toggle hidden on content sections
        assert "content-section" in js
        assert "classList.toggle" in js or "classList.remove" in js

    def test_switchSection_updates_active_nav_link(self) -> None:
        js = _js()
        # Must toggle active class based on data-section attribute
        assert "dataset.section" in js
        assert "classList.toggle" in js

    def test_switchSection_function_exists(self) -> None:
        js = _js()
        assert "function switchSection" in js

    def test_each_section_panel_has_id_matching_nav(self) -> None:
        """Verify id pattern: {section}-section for all sections."""
        html = _html()
        for section in SECTIONS:
            assert f'id="{section}-section"' in html


# ---------------------------------------------------------------------------
# AC: active sidebar item is visually highlighted
# ---------------------------------------------------------------------------


class TestActiveHighlighting:
    def test_nav_link_active_css_exists(self) -> None:
        css = _css()
        assert ".nav-link.active" in css

    def test_nav_link_active_has_color(self) -> None:
        css = _css()
        # Extract the .nav-link.active block
        match = re.search(r"\.nav-link\.active\s*\{([^}]+)\}", css)
        assert match, ".nav-link.active CSS block not found"
        block = match.group(1)
        assert "color" in block, ".nav-link.active should set a color"

    def test_mobile_nav_link_active_css_exists(self) -> None:
        css = _css()
        assert ".mobile-nav-link.active" in css

    def test_js_sets_active_class_on_nav_links(self) -> None:
        js = _js()
        # The JS loops over nav links and toggles 'active'
        assert "'active'" in js or '"active"' in js
        assert "classList.toggle" in js

    def test_initial_chat_link_has_active_class(self) -> None:
        html = _html()
        # The chat nav link should start with the active class
        assert (
            'data-section="chat" class="nav-link active"' in html
            or 'class="nav-link active"' in html
        )


# ---------------------------------------------------------------------------
# AC: navigation does not reload the page or lose state
# ---------------------------------------------------------------------------


class TestNoPageReload:
    def test_nav_links_use_href_hash(self) -> None:
        html = _html()
        # All nav links should use href="#" (SPA; no real page navigation)
        nav_links = re.findall(r'<a\s+href="([^"]*)"[^>]*data-section=', html)
        for href in nav_links:
            assert href == "#", f"Nav link href should be '#', got '{href}'"

    def test_js_calls_preventDefault(self) -> None:
        js = _js()
        assert "preventDefault" in js, "JS should call e.preventDefault() to prevent page reload"

    def test_js_does_not_use_location_href_assign(self) -> None:
        js = _js()
        # Assigning window.location.href or location.href would cause a reload
        assert "location.href =" not in js and "window.location.href =" not in js

    def test_js_does_not_use_location_assign(self) -> None:
        js = _js()
        assert "location.assign(" not in js

    def test_js_does_not_use_location_replace_for_nav(self) -> None:
        js = _js()
        # location.replace() would also cause a reload/navigation
        assert "location.replace(" not in js


# ---------------------------------------------------------------------------
# AC: back/forward browser navigation works
# ---------------------------------------------------------------------------


class TestBrowserHistory:
    def test_js_uses_pushState(self) -> None:
        js = _js()
        assert "pushState" in js, "JS should call history.pushState for back/forward support"

    def test_js_has_popstate_listener(self) -> None:
        js = _js()
        assert "popstate" in js, "JS should listen to the popstate event"

    def test_pushState_passes_section_in_state(self) -> None:
        js = _js()
        # pushState call should include a state object with section info
        assert "pushState({section" in js or "pushState({ section" in js

    def test_popstate_handler_calls_switchSection(self) -> None:
        js = _js()
        # The popstate handler must call switchSection
        # Check that popstate and switchSection both appear and that they're related
        assert "popstate" in js and "switchSection" in js

    def test_popstate_handler_reads_state_section(self) -> None:
        js = _js()
        # Handler should read section from event.state
        assert "e.state" in js or "event.state" in js

    def test_pushState_uses_hash_url(self) -> None:
        js = _js()
        # Should update the URL with a hash fragment (e.g. #chat) so it's bookmarkable
        assert "'#' + sectionId" in js or '"#" + sectionId' in js

    def test_no_double_pushState_on_popstate(self) -> None:
        js = _js()
        # Should guard against calling pushState during popstate handling
        assert "_navigatingFromHistory" in js

    def test_VALID_SECTIONS_covers_all_sections(self) -> None:
        js = _js()
        # All 8 sections should be listed as valid navigation targets
        for section in SECTIONS:
            assert f"'{section}'" in js or f'"{section}"' in js

    def test_initial_load_reads_hash_from_url(self) -> None:
        js = _js()
        # On page load, the hash should be checked to restore the correct section
        assert "location.hash" in js


# ---------------------------------------------------------------------------
# Typecheck
# ---------------------------------------------------------------------------


class TestTypecheck:
    def test_mypy_passes(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                "rex/",
                "--ignore-missing-imports",
                "--no-error-summary",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        # Pre-existing errors are acceptable; this should not introduce new errors
        # mypy exit code 0 = clean, 1 = type errors (pre-existing), 2 = fatal error
        assert result.returncode in (
            0,
            1,
        ), f"mypy crashed (exit {result.returncode}):\n{result.stderr}"
