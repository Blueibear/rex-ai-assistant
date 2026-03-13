"""Tests for US-172: Typing / thinking indicator in chat during LLM generation."""

from pathlib import Path

STATIC_DIR = Path(__file__).parent.parent / "rex" / "dashboard" / "static"
CSS_FILE = STATIC_DIR / "css" / "dashboard.css"
JS_FILE = STATIC_DIR / "js" / "dashboard.js"


def _css() -> str:
    return CSS_FILE.read_text(encoding="utf-8")


def _js() -> str:
    return JS_FILE.read_text(encoding="utf-8")


def _get_handle_chat_body() -> str:
    js = _js()
    idx = js.index("async function handleChatSubmit")
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


# ── CSS styling ────────────────────────────────────────────────────────────────


class TestCSSThinkingIndicator:
    def test_thinking_class_exists(self):
        assert ".chat-message.thinking" in _css()

    def test_thinking_dots_class_exists(self):
        assert ".thinking-dots" in _css()

    def test_thinking_dot_class_exists(self):
        assert ".thinking-dot" in _css()

    def test_thinking_dot_has_animation(self):
        css = _css()
        idx = css.index(".thinking-dot {")
        block_end = css.index("}", idx)
        block = css[idx:block_end]
        assert "animation" in block

    def test_thinking_bounce_keyframes_exist(self):
        assert "@keyframes thinking-bounce" in _css()

    def test_thinking_keyframes_has_transform(self):
        css = _css()
        idx = css.index("@keyframes thinking-bounce")
        brace_start = css.index("{", idx)
        depth = 0
        i = brace_start
        while i < len(css):
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
                if depth == 0:
                    body = css[brace_start : i + 1]
                    break
            i += 1
        assert "transform" in body

    def test_thinking_dots_is_flex(self):
        css = _css()
        idx = css.index(".thinking-dots {")
        block_end = css.index("}", idx)
        block = css[idx:block_end]
        assert "flex" in block

    def test_thinking_dot_staggered_delays(self):
        css = _css()
        assert "animation-delay" in css

    def test_thinking_dot_is_circular(self):
        css = _css()
        idx = css.index(".thinking-dot {")
        block_end = css.index("}", idx)
        block = css[idx:block_end]
        assert "border-radius" in block


# ── JS logic ──────────────────────────────────────────────────────────────────


class TestJSThinkingIndicator:
    def test_thinking_indicator_id_in_chat_submit(self):
        body = _get_handle_chat_body()
        assert "thinking-indicator" in body

    def test_thinking_indicator_added_before_fetch(self):
        body = _get_handle_chat_body()
        indicator_pos = body.index("thinking-indicator")
        fetch_pos = body.index("fetch(")
        assert indicator_pos < fetch_pos

    def test_thinking_dots_class_in_indicator_html(self):
        body = _get_handle_chat_body()
        assert "thinking-dots" in body

    def test_thinking_dot_spans_in_indicator_html(self):
        body = _get_handle_chat_body()
        assert "thinking-dot" in body

    def test_indicator_removed_on_first_token(self):
        body = _get_handle_chat_body()
        # thinking.remove() must happen inside the token event handler
        assert "thinking" in body
        assert ".remove()" in body

    def test_indicator_handled_on_error(self):
        body = _get_handle_chat_body()
        # Error path must reference thinking-indicator
        # Error is in catch block
        catch_idx = body.rindex("catch")
        catch_body = body[catch_idx:]
        assert "thinking" in catch_body

    def test_thinking_class_removed_on_error(self):
        body = _get_handle_chat_body()
        catch_idx = body.rindex("catch")
        catch_body = body[catch_idx:]
        assert "classList" in catch_body or "textContent" in catch_body
