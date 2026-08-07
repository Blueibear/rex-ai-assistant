"""Tests for US-165: Loading and error states for all data-fetching panels.

RETIRED: The rex/dashboard/ module was retired in OpenClaw migration iteration 93
(US-P7-014). These tests depended on dashboard static assets that no longer exist.
All tests in this file are skipped.
"""

import pytest

pytestmark = pytest.mark.skip(reason="rex/dashboard retired in OpenClaw migration (US-P7-014)")


class TestCSSLoadingError:
    def test_placeholder(self):
        pass


class TestJSRetryHelper:
    def test_placeholder(self):
        pass


class TestLoadingIndicators:
    def test_placeholder(self):
        pass
