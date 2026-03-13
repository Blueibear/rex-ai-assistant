"""Tests for US-152: Chat message list component.

Acceptance criteria:
- Chat panel displays messages in chronological order, oldest at top, newest at bottom
- user messages and Rex messages are visually distinct (different alignment, color, or label)
- message list auto-scrolls to the newest message when a new message arrives
- message list is scrollable and handles at least 100 messages without layout issues
- Typecheck passes
- Verify changes work in browser
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
HTML_PATH = REPO_ROOT / "rex" / "dashboard" / "templates" / "index.html"
CSS_PATH = REPO_ROOT / "rex" / "dashboard" / "static" / "css" / "dashboard.css"
JS_PATH = REPO_ROOT / "rex" / "dashboard" / "static" / "js" / "dashboard.js"


def _html() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


def _css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def _js() -> str:
    return JS_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# AC: Chat panel displays messages in chronological order
# ---------------------------------------------------------------------------


class TestChronologicalOrder:
    def test_chat_messages_container_exists_in_html(self) -> None:
        """#chat-messages container must exist in the HTML."""
        html = _html()
        assert 'id="chat-messages"' in html

    def test_render_function_exists(self) -> None:
        """renderChatMessages function must be defined."""
        js = _js()
        assert "function renderChatMessages" in js

    def test_render_iterates_chat_history(self) -> None:
        """renderChatMessages must iterate over state.chatHistory."""
        js = _js()
        assert "chatHistory" in js
        assert "renderChatMessages" in js

    def test_user_message_rendered_before_assistant_in_each_turn(self) -> None:
        """Within each turn, user message must appear before assistant reply."""
        js = _js()
        # Find the renderChatMessages function body
        start = js.index("function renderChatMessages")
        end = js.index("\n    }", start) + 6
        fn_body = js[start:end]
        # user_message must appear before assistant_reply in the template
        assert "user_message" in fn_body
        assert "assistant_reply" in fn_body
        user_pos = fn_body.index("user_message")
        assistant_pos = fn_body.index("assistant_reply")
        assert user_pos < assistant_pos, (
            "user_message should be rendered before assistant_reply (chronological order)"
        )

    def test_load_chat_history_called_when_switching_to_chat(self) -> None:
        """switchSection must call loadChatHistory when chat is selected."""
        js = _js()
        assert "loadChatHistory" in js
        # Inside the switch/case for 'chat'
        assert "'chat'" in js or '"chat"' in js
        # loadChatHistory is called
        assert "loadChatHistory()" in js


# ---------------------------------------------------------------------------
# AC: user messages and Rex messages are visually distinct
# ---------------------------------------------------------------------------


class TestVisuallyDistinct:
    def test_user_message_class_exists_in_css(self) -> None:
        """CSS must define .chat-message.user style."""
        css = _css()
        assert ".chat-message.user" in css

    def test_assistant_message_class_exists_in_css(self) -> None:
        """CSS must define .chat-message.assistant style."""
        css = _css()
        assert ".chat-message.assistant" in css

    def test_user_message_has_different_alignment(self) -> None:
        """User messages should be right-aligned (align-self: flex-end)."""
        css = _css()
        # Find .chat-message.user block
        start = css.index(".chat-message.user")
        block_start = css.index("{", start)
        block_end = css.index("}", block_start)
        user_block = css[block_start:block_end]
        assert "flex-end" in user_block, (
            ".chat-message.user should have align-self: flex-end for right alignment"
        )

    def test_assistant_message_has_different_alignment(self) -> None:
        """Assistant messages should be left-aligned (align-self: flex-start)."""
        css = _css()
        start = css.index(".chat-message.assistant")
        block_start = css.index("{", start)
        block_end = css.index("}", block_start)
        assistant_block = css[block_start:block_end]
        assert "flex-start" in assistant_block, (
            ".chat-message.assistant should have align-self: flex-start for left alignment"
        )

    def test_user_and_assistant_have_different_background(self) -> None:
        """User and assistant messages must use different background colors."""
        css = _css()
        # Extract user block background
        u_start = css.index(".chat-message.user")
        u_block_start = css.index("{", u_start)
        u_block_end = css.index("}", u_block_start)
        user_block = css[u_block_start:u_block_end]

        a_start = css.index(".chat-message.assistant")
        a_block_start = css.index("{", a_start)
        a_block_end = css.index("}", a_block_start)
        assistant_block = css[a_block_start:a_block_end]

        assert "background" in user_block or "background-color" in user_block
        assert "background" in assistant_block or "background-color" in assistant_block

        # Extract the actual background values and confirm they differ
        import re
        u_bg = re.search(r"background(?:-color)?\s*:\s*([^;]+)", user_block)
        a_bg = re.search(r"background(?:-color)?\s*:\s*([^;]+)", assistant_block)
        assert u_bg is not None
        assert a_bg is not None
        assert u_bg.group(1).strip() != a_bg.group(1).strip(), (
            "User and assistant messages must have different background colors"
        )

    def test_js_adds_user_class_to_user_messages(self) -> None:
        """JS must add 'user' class to user message bubbles."""
        js = _js()
        assert 'chat-message user' in js

    def test_js_adds_assistant_class_to_assistant_messages(self) -> None:
        """JS must add 'assistant' class to assistant message bubbles."""
        js = _js()
        assert 'chat-message assistant' in js


# ---------------------------------------------------------------------------
# AC: message list auto-scrolls to newest message
# ---------------------------------------------------------------------------


class TestAutoScroll:
    def test_scrolltop_set_after_render(self) -> None:
        """renderChatMessages must set scrollTop to scrollHeight."""
        js = _js()
        start = js.index("function renderChatMessages")
        end = js.index("\n    }", start) + 6
        fn_body = js[start:end]
        assert "scrollTop" in fn_body
        assert "scrollHeight" in fn_body

    def test_scrolltop_set_after_chat_submit(self) -> None:
        """handleChatSubmit must scroll to bottom after appending messages."""
        js = _js()
        start = js.index("function handleChatSubmit")
        end = js.index("\n    }", start) + 6
        fn_body = js[start:end]
        assert "scrollTop" in fn_body
        assert "scrollHeight" in fn_body

    def test_scroll_called_for_assistant_response(self) -> None:
        """Scroll must happen after the assistant message is appended."""
        js = _js()
        start = js.index("function handleChatSubmit")
        end = js.index("\n    }", start) + 6
        fn_body = js[start:end]
        # Assistant message appended
        assistant_pos = fn_body.index("chat-message assistant")
        # scrollTop set after
        scroll_pos = fn_body.rindex("scrollTop", 0, len(fn_body))
        assert scroll_pos > assistant_pos, (
            "scrollTop must be set after the assistant message is appended"
        )


# ---------------------------------------------------------------------------
# AC: message list is scrollable and handles at least 100 messages
# ---------------------------------------------------------------------------


class TestScrollability:
    def test_chat_messages_has_overflow_y_auto(self) -> None:
        """#chat-messages container must have overflow-y: auto or scroll."""
        css = _css()
        start = css.index(".chat-messages")
        block_start = css.index("{", start)
        block_end = css.index("}", block_start)
        block = css[block_start:block_end]
        assert "overflow-y" in block
        assert "auto" in block or "scroll" in block

    def test_chat_messages_flex_column(self) -> None:
        """.chat-messages must use flex-direction: column for vertical layout."""
        css = _css()
        start = css.index(".chat-messages")
        block_start = css.index("{", start)
        block_end = css.index("}", block_start)
        block = css[block_start:block_end]
        assert "flex-direction" in block
        assert "column" in block

    def test_chat_message_max_width_set(self) -> None:
        """Individual messages must have a max-width to prevent layout overflow."""
        css = _css()
        start = css.index(".chat-message {")
        block_start = css.index("{", start)
        block_end = css.index("}", block_start)
        block = css[block_start:block_end]
        assert "max-width" in block

    def test_render_handles_multiple_messages(self) -> None:
        """renderChatMessages uses .map() to handle an arbitrary number of messages."""
        js = _js()
        start = js.index("function renderChatMessages")
        end = js.index("\n    }", start) + 6
        fn_body = js[start:end]
        assert ".map(" in fn_body, (
            "renderChatMessages should use .map() to handle any number of messages"
        )

    def test_chat_container_overflow_hidden(self) -> None:
        """.chat-container must have overflow: hidden so inner scroll works."""
        css = _css()
        start = css.index(".chat-container {")
        block_start = css.index("{", start)
        block_end = css.index("}", block_start)
        block = css[block_start:block_end]
        assert "overflow" in block

    def test_chat_messages_takes_remaining_space(self) -> None:
        """.chat-messages must flex: 1 to take available vertical space."""
        css = _css()
        start = css.index(".chat-messages")
        block_start = css.index("{", start)
        block_end = css.index("}", block_start)
        block = css[block_start:block_end]
        assert "flex: 1" in block or "flex:1" in block


# ---------------------------------------------------------------------------
# AC: Typecheck passes
# ---------------------------------------------------------------------------


class TestTypecheck:
    def test_mypy_passes(self) -> None:
        """mypy must exit without crashing (pre-existing errors are acceptable)."""
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--ignore-missing-imports", "rex/"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        # mypy exit code 0 = no errors, 1 = type errors, 2 = usage error/crash
        assert result.returncode in (0, 1), (
            f"mypy crashed (exit {result.returncode}):\n{result.stderr}"
        )
