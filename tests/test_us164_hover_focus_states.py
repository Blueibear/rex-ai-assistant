"""Tests for US-164: Consistent component styling and hover/focus states."""

from pathlib import Path

STATIC_DIR = Path(__file__).parent.parent / "rex" / "dashboard" / "static"
CSS_FILE = STATIC_DIR / "css" / "dashboard.css"


def _css() -> str:
    return CSS_FILE.read_text(encoding="utf-8")


def _block_for(css: str, selector: str) -> str:
    """Return the CSS block (between braces) for the first occurrence of selector."""
    idx = css.index(selector)
    brace_start = css.index("{", idx)
    depth = 0
    i = brace_start
    while i < len(css):
        if css[i] == "{":
            depth += 1
        elif css[i] == "}":
            depth -= 1
            if depth == 0:
                return css[brace_start : i + 1]
        i += 1
    return css[brace_start:]


# ── Design tokens ──────────────────────────────────────────────────────────────


class TestDesignTokens:
    def test_focus_ring_token_defined(self):
        assert "--focus-ring" in _css()

    def test_danger_hover_token_defined(self):
        assert "--danger-hover" in _css()

    def test_focus_ring_token_used_in_btn_focus(self):
        css = _css()
        assert ".btn:focus-visible" in css
        block = _block_for(css, ".btn:focus-visible")
        assert "var(--focus-ring)" in block

    def test_focus_ring_token_used_in_input_focus(self):
        css = _css()
        # At least one input focus rule must use the token
        assert "var(--focus-ring)" in css


# ── Button focus/hover ─────────────────────────────────────────────────────────


class TestButtonFocusHover:
    def test_btn_focus_visible_defined(self):
        assert ".btn:focus-visible" in _css()

    def test_btn_focus_has_outline(self):
        css = _css()
        block = _block_for(css, ".btn:focus-visible")
        assert "outline" in block

    def test_btn_focus_outline_uses_primary_color(self):
        css = _css()
        block = _block_for(css, ".btn:focus-visible")
        assert "var(--primary-color)" in block

    def test_btn_focus_has_box_shadow(self):
        css = _css()
        block = _block_for(css, ".btn:focus-visible")
        assert "box-shadow" in block

    def test_btn_primary_hover_exists(self):
        assert ".btn-primary:hover" in _css()

    def test_btn_primary_hover_uses_token(self):
        css = _css()
        block = _block_for(css, ".btn-primary:hover")
        assert "var(--primary-hover)" in block

    def test_btn_secondary_hover_exists(self):
        assert ".btn-secondary:hover" in _css()

    def test_btn_danger_hover_exists(self):
        assert ".btn-danger:hover" in _css()

    def test_btn_danger_hover_uses_token(self):
        css = _css()
        block = _block_for(css, ".btn-danger:hover")
        assert "var(--danger-hover)" in block

    def test_btn_focus_outline_offset_set(self):
        css = _css()
        block = _block_for(css, ".btn:focus-visible")
        assert "outline-offset" in block


# ── Input focus ────────────────────────────────────────────────────────────────


class TestInputFocus:
    def test_input_focus_visible_rule_exists(self):
        assert "input:focus-visible" in _css()

    def test_select_focus_visible_rule_exists(self):
        assert "select:focus-visible" in _css()

    def test_textarea_focus_visible_rule_exists(self):
        assert "textarea:focus-visible" in _css()

    def test_input_focus_visible_has_outline(self):
        css = _css()
        block = _block_for(css, "input:focus-visible")
        assert "outline" in block

    def test_input_focus_visible_uses_primary_color(self):
        css = _css()
        block = _block_for(css, "input:focus-visible")
        assert "var(--primary-color)" in block

    def test_form_group_input_focus_uses_token_shadow(self):
        css = _css()
        block = _block_for(css, ".form-group input:focus")
        assert "var(--focus-ring)" in block

    def test_chat_input_focus_has_box_shadow(self):
        css = _css()
        block = _block_for(css, ".chat-form input:focus")
        assert "box-shadow" in block

    def test_notif_filter_select_focus_has_box_shadow(self):
        css = _css()
        block = _block_for(css, ".notif-filter-select:focus")
        assert "box-shadow" in block


# ── Link focus ─────────────────────────────────────────────────────────────────


class TestLinkFocus:
    def test_a_focus_visible_rule_exists(self):
        assert "a:focus-visible" in _css()

    def test_a_focus_visible_has_outline(self):
        css = _css()
        block = _block_for(css, "a:focus-visible")
        assert "outline" in block

    def test_a_focus_visible_uses_primary_color(self):
        css = _css()
        block = _block_for(css, "a:focus-visible")
        assert "var(--primary-color)" in block
