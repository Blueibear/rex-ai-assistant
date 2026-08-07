"""Tests for US-151: Implement active navigation state and panel switching.

RETIRED: The rex/dashboard/ module was retired in OpenClaw migration iteration 93
(US-P7-014). These tests depended on dashboard static assets that no longer exist.
All tests in this file are skipped.
"""

import pytest

pytestmark = pytest.mark.skip(reason="rex/dashboard retired in OpenClaw migration (US-P7-014)")


class TestSidebarPanelSwitching:
    def test_placeholder(self):
        pass


class TestActiveHighlighting:
    def test_placeholder(self):
        pass


class TestNoPageReload:
    def test_placeholder(self):
        pass
