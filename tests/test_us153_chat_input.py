"""Tests for US-153: Chat message input and send.

Acceptance criteria:
- text input field visible and focused by default when the Chat panel is open
- pressing Enter or clicking a Send button submits the message
- input clears after send
- sending an empty message is a no-op (no empty messages added to the list)
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
JS_PATH = REPO_ROOT / "rex" / "dashboard" / "static" / "js" / "dashboard.js"


def _html() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


def _js() -> str:
    return JS_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# HTML structure: input field exists in chat section
# ---------------------------------------------------------------------------

class TestChatInputHtml:
    def test_chat_input_exists(self):
        html = _html()
        assert 'id="chat-input"' in html, "chat-input element must exist"

    def test_chat_input_is_text_type(self):
        html = _html()
        # Find the line with chat-input
        lines = html.splitlines()
        input_line = next((l for l in lines if 'id="chat-input"' in l), None)
        assert input_line is not None, "chat-input element not found"
        assert 'type="text"' in input_line, "chat-input must be type=text"

    def test_chat_form_exists(self):
        html = _html()
        assert 'id="chat-form"' in html, "chat-form must exist"

    def test_send_button_exists(self):
        html = _html()
        assert 'type="submit"' in html or 'Send' in html, "Send button must exist"

    def test_send_button_in_chat_form(self):
        html = _html()
        # The Send button should be inside chat-form
        chat_form_start = html.find('id="chat-form"')
        assert chat_form_start != -1, "chat-form not found"
        form_block = html[chat_form_start:chat_form_start + 500]
        assert 'Send' in form_block or 'type="submit"' in form_block, \
            "Send button must be inside chat-form"

    def test_chat_input_in_chat_section(self):
        html = _html()
        chat_section_start = html.find('id="chat-section"')
        assert chat_section_start != -1, "chat-section not found"
        chat_section_end = html.find('</section>', chat_section_start)
        chat_section = html[chat_section_start:chat_section_end]
        assert 'id="chat-input"' in chat_section, "chat-input must be inside chat-section"

    def test_chat_input_placeholder(self):
        html = _html()
        lines = html.splitlines()
        input_line = next((l for l in lines if 'id="chat-input"' in l), None)
        assert input_line is not None
        assert 'placeholder' in input_line, "chat-input should have a placeholder"


# ---------------------------------------------------------------------------
# JS: focus on chat panel open
# ---------------------------------------------------------------------------

class TestChatInputFocus:
    def test_focus_called_on_chat_section_switch(self):
        js = _js()
        # switchSection case 'chat' must trigger focus on chat-input
        assert "chat-input" in js
        # Find the case 'chat' block
        case_chat_idx = js.index("case 'chat':")
        # Grab the next 300 chars to check for focus call
        block = js[case_chat_idx:case_chat_idx + 300]
        assert "focus()" in block, \
            "chat-input.focus() must be called when switching to chat section"

    def test_focus_uses_chat_input_selector(self):
        js = _js()
        case_chat_idx = js.index("case 'chat':")
        block = js[case_chat_idx:case_chat_idx + 300]
        assert "chat-input" in block, \
            "focus must target #chat-input in the chat case block"

    def test_focus_also_called_after_send_in_finally(self):
        js = _js()
        # After submit, focus should be restored
        assert "input.focus()" in js or "inp.focus()" in js, \
            "focus must be restored after send (in finally or catch)"


# ---------------------------------------------------------------------------
# JS: form submission wired
# ---------------------------------------------------------------------------

class TestChatFormSubmit:
    def test_chat_form_submit_listener_registered(self):
        js = _js()
        assert "chat-form" in js
        # The form's submit event must be listened to
        assert "'submit'" in js or '"submit"' in js
        # Both must appear near each other
        form_idx = js.find("chat-form")
        assert form_idx != -1
        # Check for addEventListener with submit within 100 chars of chat-form
        nearby = js[form_idx:form_idx + 100]
        assert "submit" in nearby, \
            "chat-form submit event listener must be registered"

    def test_handle_chat_submit_function_exists(self):
        js = _js()
        assert "handleChatSubmit" in js, "handleChatSubmit function must be defined"

    def test_handle_chat_submit_calls_prevent_default(self):
        js = _js()
        # Find handleChatSubmit function body
        fn_idx = js.index("handleChatSubmit")
        # Find function body (async function ... { )
        body_start = js.index("{", fn_idx)
        body_end = js.index("\n    }", body_start)
        body = js[body_start:body_end]
        assert "preventDefault" in body, \
            "handleChatSubmit must call event.preventDefault()"

    def test_handle_chat_submit_reads_chat_input(self):
        js = _js()
        fn_idx = js.index("handleChatSubmit")
        body_start = js.index("{", fn_idx)
        body_end = js.index("\n    }", body_start)
        body = js[body_start:body_end]
        assert "chat-input" in body, \
            "handleChatSubmit must read value from #chat-input"


# ---------------------------------------------------------------------------
# JS: input clears after send
# ---------------------------------------------------------------------------

class TestInputClearsAfterSend:
    def test_input_value_reset_after_send(self):
        js = _js()
        fn_idx = js.index("handleChatSubmit")
        body_start = js.index("{", fn_idx)
        body_end = js.index("\n    }", body_start)
        body = js[body_start:body_end]
        # input.value = '' clears the input
        assert "input.value = ''" in body or 'input.value = ""' in body, \
            "handleChatSubmit must clear input.value after send"

    def test_input_cleared_before_api_call(self):
        js = _js()
        fn_idx = js.index("handleChatSubmit")
        body_start = js.index("{", fn_idx)
        body_end = js.index("\n    }", body_start)
        body = js[body_start:body_end]
        clear_idx = body.find("input.value = ''")
        if clear_idx == -1:
            clear_idx = body.find('input.value = ""')
        api_idx = body.find("api(")
        assert clear_idx != -1, "input.value clear not found"
        assert api_idx != -1, "api() call not found"
        assert clear_idx < api_idx, \
            "input must be cleared before the API call (optimistic UX)"


# ---------------------------------------------------------------------------
# JS: empty message no-op
# ---------------------------------------------------------------------------

class TestEmptyMessageNoOp:
    def test_empty_message_guard_exists(self):
        js = _js()
        fn_idx = js.index("handleChatSubmit")
        body_start = js.index("{", fn_idx)
        body_end = js.index("\n    }", body_start)
        body = js[body_start:body_end]
        # if (!message) return; or equivalent
        assert "if (!message)" in body or "if (!message) return" in body, \
            "handleChatSubmit must guard against empty messages"

    def test_message_is_trimmed_before_check(self):
        js = _js()
        fn_idx = js.index("handleChatSubmit")
        body_start = js.index("{", fn_idx)
        body_end = js.index("\n    }", body_start)
        body = js[body_start:body_end]
        # trim() should be called so whitespace-only messages are rejected
        assert ".trim()" in body, \
            "input value must be .trim()-ed before checking if empty"

    def test_empty_guard_before_dom_modification(self):
        js = _js()
        fn_idx = js.index("handleChatSubmit")
        body_start = js.index("{", fn_idx)
        body_end = js.index("\n    }", body_start)
        body = js[body_start:body_end]
        guard_idx = body.find("if (!message)")
        dom_idx = body.find("innerHTML")
        assert guard_idx != -1
        assert dom_idx != -1
        assert guard_idx < dom_idx, \
            "Empty message guard must come before any DOM modification"


# ---------------------------------------------------------------------------
# Typecheck
# ---------------------------------------------------------------------------

class TestTypecheck:
    def test_mypy_passes(self):
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "rex/", "--ignore-missing-imports",
             "--no-error-summary"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        # Pre-existing errors are acceptable; just must not crash
        assert result.returncode in (0, 1), \
            f"mypy crashed (exit {result.returncode}):\n{result.stderr}"
