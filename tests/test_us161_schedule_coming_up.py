"""Tests for US-161: Upcoming events and due items panel section."""

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


class TestHTMLComingUp:
    def test_coming_up_list_container_exists(self):
        assert 'id="schedule-coming-up-list"' in _html()

    def test_coming_up_section_exists(self):
        assert "schedule-coming-up" in _html()

    def test_coming_up_heading_exists(self):
        assert "Coming Up" in _html()

    def test_coming_up_empty_state_exists(self):
        html = _html()
        assert "Nothing due in the next 24 hours" in html

    def test_coming_up_inside_schedule_section(self):
        html = _html()
        section_start = html.index('id="schedule-section"')
        assert html.index('id="schedule-coming-up-list"') > section_start

    def test_coming_up_before_schedule_list(self):
        html = _html()
        coming_up_idx = html.index('id="schedule-coming-up-list"')
        list_idx = html.index('id="schedule-list"')
        assert coming_up_idx < list_idx


# ── CSS styling ────────────────────────────────────────────────────────────────


class TestCSSComingUp:
    def test_coming_up_section_class_defined(self):
        assert ".schedule-coming-up" in _css()

    def test_coming_up_heading_class_defined(self):
        assert ".schedule-coming-up-heading" in _css()

    def test_coming_up_list_class_defined(self):
        assert ".schedule-coming-up-list" in _css()

    def test_coming_up_item_class_defined(self):
        assert ".schedule-coming-up-item" in _css()

    def test_coming_up_name_class_defined(self):
        assert ".schedule-coming-up-name" in _css()

    def test_coming_up_time_class_defined(self):
        assert ".schedule-coming-up-time" in _css()

    def test_coming_up_list_is_flex(self):
        css = _css()
        idx = css.index(".schedule-coming-up-list")
        block_end = css.index("}", idx)
        block = css[idx:block_end]
        assert "flex" in block

    def test_coming_up_item_has_justify_content(self):
        css = _css()
        idx = css.index(".schedule-coming-up-item")
        block_end = css.index("}", idx)
        block = css[idx:block_end]
        assert "justify-content" in block


# ── JS logic ──────────────────────────────────────────────────────────────────


class TestJSComingUp:
    def _get_render_coming_up_body(self) -> str:
        js = _js()
        idx = js.index("function renderComingUp")
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

    def _get_load_fn_body(self) -> str:
        js = _js()
        idx = js.index("async function loadScheduleJobs")
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

    def test_render_coming_up_fn_exists(self):
        assert "renderComingUp" in _js()

    def test_time_until_helper_exists(self):
        assert "_timeUntil" in _js()

    def test_load_fn_calls_render_coming_up(self):
        assert "renderComingUp" in self._get_load_fn_body()

    def test_render_coming_up_targets_container(self):
        body = self._get_render_coming_up_body()
        assert "schedule-coming-up-list" in body

    def test_render_coming_up_filters_24h(self):
        body = self._get_render_coming_up_body()
        assert "24" in body

    def test_render_coming_up_sorts_by_next_run(self):
        body = self._get_render_coming_up_body()
        assert "sort" in body
        assert "next_run" in body

    def test_render_coming_up_shows_time_until(self):
        body = self._get_render_coming_up_body()
        assert "_timeUntil" in body or "timeStr" in body

    def test_render_coming_up_shows_item_name(self):
        body = self._get_render_coming_up_body()
        assert "schedule-coming-up-name" in body

    def test_render_coming_up_shows_item_time(self):
        body = self._get_render_coming_up_body()
        assert "schedule-coming-up-time" in body

    def test_render_coming_up_empty_state_message(self):
        body = self._get_render_coming_up_body()
        assert "Nothing due" in body or "schedule-empty" in body

    def test_time_until_uses_hours(self):
        js = _js()
        idx = js.index("function _timeUntil")
        brace_start = js.index("{", idx)
        depth = 0
        i = brace_start
        while i < len(js):
            if js[i] == "{":
                depth += 1
            elif js[i] == "}":
                depth -= 1
                if depth == 0:
                    fn_body = js[brace_start : i + 1]
                    break
            i += 1
        assert "hour" in fn_body
        assert "minute" in fn_body

    def test_render_coming_up_uses_escape_html(self):
        body = self._get_render_coming_up_body()
        assert "escapeHtml" in body
