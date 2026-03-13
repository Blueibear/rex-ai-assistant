"""Tests for US-166: Responsive layout for common window sizes."""

from pathlib import Path

STATIC_DIR = Path(__file__).parent.parent / "rex" / "dashboard" / "static"
TEMPLATE_DIR = Path(__file__).parent.parent / "rex" / "dashboard" / "templates"
CSS_FILE = STATIC_DIR / "css" / "dashboard.css"
HTML_FILE = TEMPLATE_DIR / "index.html"


def _css() -> str:
    return CSS_FILE.read_text(encoding="utf-8")


def _html() -> str:
    return HTML_FILE.read_text(encoding="utf-8")


# ── HTML structure ─────────────────────────────────────────────────────────────


class TestHTMLResponsive:
    def test_nav_label_spans_exist(self):
        assert "nav-label" in _html()

    def test_nav_label_wraps_chat(self):
        html = _html()
        assert '<span class="nav-label">Chat</span>' in html

    def test_nav_label_wraps_voice(self):
        assert '<span class="nav-label">Voice</span>' in _html()

    def test_nav_label_wraps_schedule(self):
        assert '<span class="nav-label">Schedule</span>' in _html()

    def test_nav_label_wraps_settings(self):
        assert '<span class="nav-label">Settings</span>' in _html()

    def test_mobile_nav_exists(self):
        assert "mobile-nav" in _html()


# ── CSS responsive rules ───────────────────────────────────────────────────────


class TestCSSResponsive:
    def test_body_has_overflow_x_hidden(self):
        css = _css()
        # Find body rule and check overflow-x
        idx = css.index("html, body {")
        block_end = css.index("}", idx)
        block = css[idx:block_end]
        assert "overflow-x: hidden" in block

    def test_media_query_1024px_exists(self):
        assert "max-width: 1024px" in _css()

    def test_media_query_768px_exists(self):
        assert "max-width: 768px" in _css()

    def test_1024px_breakpoint_collapses_sidebar(self):
        css = _css()
        idx = css.index("max-width: 1024px")
        block_start = css.index("{", idx)
        depth = 0
        i = block_start
        while i < len(css):
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
                if depth == 0:
                    media_block = css[block_start : i + 1]
                    break
            i += 1
        # Sidebar should be narrowed (64px) or nav labels hidden
        assert "nav-label" in media_block or "sidebar-width" in media_block

    def test_1024px_hides_nav_labels(self):
        css = _css()
        idx = css.index("max-width: 1024px")
        block_start = css.index("{", idx)
        depth = 0
        i = block_start
        while i < len(css):
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
                if depth == 0:
                    media_block = css[block_start : i + 1]
                    break
            i += 1
        assert ".nav-label" in media_block

    def test_768px_hides_sidebar(self):
        css = _css()
        idx = css.index("max-width: 768px")
        # Find the opening brace of this media block
        block_start = css.index("{", idx)
        depth = 0
        i = block_start
        while i < len(css):
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
                if depth == 0:
                    media_block = css[block_start : i + 1]
                    break
            i += 1
        assert ".sidebar" in media_block

    def test_768px_shows_mobile_nav(self):
        css = _css()
        idx = css.index("max-width: 768px")
        block_start = css.index("{", idx)
        depth = 0
        i = block_start
        while i < len(css):
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
                if depth == 0:
                    media_block = css[block_start : i + 1]
                    break
            i += 1
        assert "mobile-nav" in media_block

    def test_main_content_has_overflow_x_hidden(self):
        css = _css()
        idx = css.index(".main-content {")
        block_end = css.index("}", idx)
        block = css[idx:block_end]
        assert "overflow-x: hidden" in block

    def test_sidebar_width_token_used(self):
        # Sidebar width uses the CSS custom property
        assert "--sidebar-width" in _css()
