"""Tests for US-163: Quick action buttons on the Overview panel."""

from pathlib import Path

STATIC_DIR = Path(__file__).parent.parent / "rex" / "dashboard" / "static"
TEMPLATE_DIR = Path(__file__).parent.parent / "rex" / "dashboard" / "templates"
CSS_FILE = STATIC_DIR / "css" / "dashboard.css"
JS_FILE = STATIC_DIR / "js" / "dashboard.js"
HTML_FILE = TEMPLATE_DIR / "index.html"


def _html() -> str:
    return HTML_FILE.read_text(encoding="utf-8")


def _css() -> str:
    return CSS_FILE.read_text(encoding="utf-8")


def _js() -> str:
    return JS_FILE.read_text(encoding="utf-8")


# ── HTML structure ─────────────────────────────────────────────────────────────


class TestHTMLQuickActions:
    def test_quick_actions_section_exists(self):
        assert "overview-quick-actions" in _html()

    def test_quick_actions_row_exists(self):
        assert "overview-quick-actions-row" in _html()

    def test_start_listening_button_exists(self):
        assert "Start Listening" in _html()

    def test_open_chat_button_exists(self):
        assert "Open Chat" in _html()

    def test_view_schedule_button_exists(self):
        assert "View Schedule" in _html()

    def test_start_listening_navigates_to_voice(self):
        html = _html()
        idx = html.index("Start Listening")
        nearby = html[max(0, idx - 200) : idx + 50]
        assert "voice" in nearby

    def test_open_chat_navigates_to_chat(self):
        html = _html()
        idx = html.index("Open Chat")
        nearby = html[max(0, idx - 200) : idx + 50]
        assert "chat" in nearby

    def test_view_schedule_navigates_to_schedule(self):
        html = _html()
        idx = html.index("View Schedule")
        nearby = html[max(0, idx - 200) : idx + 50]
        assert "schedule" in nearby

    def test_buttons_use_navigator_handler(self):
        html = _html()
        section_start = html.index('id="overview-section"')
        section_end = html.index("</section>", section_start)
        section = html[section_start:section_end]
        assert "dashboardHandlers.navigateTo" in section

    def test_quick_actions_inside_overview_section(self):
        html = _html()
        section_start = html.index('id="overview-section"')
        section_end = html.index("</section>", section_start)
        section = html[section_start:section_end]
        assert "overview-quick-actions" in section

    def test_quick_actions_heading_exists(self):
        assert "Quick Actions" in _html()

    def test_at_least_three_quick_btns(self):
        assert _html().count("overview-quick-btn") >= 3


# ── CSS styling ────────────────────────────────────────────────────────────────


class TestCSSQuickActions:
    def test_quick_actions_class_defined(self):
        assert ".overview-quick-actions" in _css()

    def test_quick_actions_row_class_defined(self):
        assert ".overview-quick-actions-row" in _css()

    def test_quick_btn_class_defined(self):
        assert ".overview-quick-btn" in _css()

    def test_quick_actions_row_uses_flex(self):
        css = _css()
        idx = css.index(".overview-quick-actions-row")
        block_end = css.index("}", idx)
        block = css[idx:block_end]
        assert "flex" in block


# ── JS logic ──────────────────────────────────────────────────────────────────


class TestJSQuickActions:
    def _get_handlers_block(self) -> str:
        js = _js()
        idx = js.index("window.dashboardHandlers = {")
        brace_start = js.index("{", idx)
        depth = 0
        i = brace_start
        while i < len(js):
            if js[i] == "{":
                depth += 1
            elif js[i] == "}":
                depth -= 1
                if depth == 0:
                    return js[brace_start : i + 1]
            i += 1
        return js[brace_start:]

    def test_navigate_to_exposed_in_handlers(self):
        assert "navigateTo" in self._get_handlers_block()

    def test_navigate_to_maps_to_switch_section(self):
        handlers = self._get_handlers_block()
        assert "switchSection" in handlers or "navigateTo: switchSection" in handlers
