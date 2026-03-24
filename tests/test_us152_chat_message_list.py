"""Tests for US-152: Chat message list component.

RETIRED: The rex/dashboard/ module was retired in OpenClaw migration iteration 93
(US-P7-014). These tests depended on dashboard static assets that no longer exist.
All tests in this file are skipped.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="rex/dashboard retired in OpenClaw migration (US-P7-014)"
)


class TestChronologicalOrder:
    def test_placeholder(self):
        pass


class TestVisuallyDistinct:
    def test_placeholder(self):
        pass


class TestAutoScroll:
    def test_placeholder(self):
        pass
