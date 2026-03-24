"""Tests for US-172: Typing / thinking indicator in chat during LLM generation.

RETIRED: The rex/dashboard/ module was retired in OpenClaw migration iteration 93
(US-P7-014).  These tests depended on dashboard static assets that no longer exist.
All tests in this file are skipped.
"""

import pytest

pytestmark = pytest.mark.skip(reason="rex/dashboard retired in OpenClaw migration (US-P7-014)")


class TestCSSThinkingIndicator:
    def test_thinking_class_exists(self):
        pass

    def test_thinking_dots_class_exists(self):
        pass


class TestJSThinkingIndicator:
    def test_thinking_indicator_id_in_chat_submit(self):
        pass
