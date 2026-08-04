"""Tests for US-149: Scaffold GUI application shell.

RETIRED: The rex/dashboard/ module was retired in OpenClaw migration iteration 93
(US-P7-014). These tests depended on dashboard static assets that no longer exist.
All tests in this file are skipped.
"""

import pytest

pytestmark = pytest.mark.skip(reason="rex/dashboard retired in OpenClaw migration (US-P7-014)")


class TestEntryPoint:
    def test_placeholder(self):
        pass


class TestSidebarNavigation:
    def test_placeholder(self):
        pass


class TestContentSections:
    def test_placeholder(self):
        pass
