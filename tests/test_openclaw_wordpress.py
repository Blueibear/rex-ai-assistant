"""Tests for US-P5-008 through US-P5-011: WordPress OpenClaw integration.

US-P5-008 acceptance criteria:
  - WordPress public API documented (audit)

US-P5-009 acceptance criteria:
  - wordpress_tool.py exists and is importable
  - TOOL_NAME = "wordpress_health_check"
  - wp_health_check() returns success dict on healthy site
  - wp_health_check() returns error dict on failure
  - register() returns None when openclaw not installed
  - ToolBridge.register_wordpress_tools() returns expected keys

US-P5-010 acceptance criteria:
  - wp_health_check() returns correct shape on success (read-through)
  - wp_health_check() forwards site_id to WordPressService.health()
  - auth_ok field is present

US-P5-011 acceptance criteria:
  - WordPress integration is read-only; no write tools exist
  - wp_health_check() is the only registered tool
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rex.wordpress.client import WPHealthResult

# ---------------------------------------------------------------------------
# US-P5-009: wordpress_tool.py — tool structure
# ---------------------------------------------------------------------------


class TestWordpressTool:
    """Basic structure and behaviour of the wordpress_tool module."""

    def test_import(self):
        from rex.openclaw.tools import wordpress_tool  # noqa: F401

    def test_tool_name(self):
        from rex.openclaw.tools.wordpress_tool import TOOL_NAME

        assert TOOL_NAME == "wordpress_health_check"

    def test_tool_description_mentions_health(self):
        from rex.openclaw.tools.wordpress_tool import TOOL_DESCRIPTION

        assert "health" in TOOL_DESCRIPTION.lower() or "check" in TOOL_DESCRIPTION.lower()

    def test_register_returns_none_without_openclaw(self):
        from rex.openclaw.tools.wordpress_tool import OPENCLAW_AVAILABLE, register

        if not OPENCLAW_AVAILABLE:
            assert register() is None
            assert register(agent=object()) is None


# ---------------------------------------------------------------------------
# US-P5-010: wp_health_check() — read path
# ---------------------------------------------------------------------------


class TestWpHealthCheck:
    """wp_health_check() correctly delegates to WordPressService.health()."""

    def _make_healthy_result(self) -> WPHealthResult:
        return WPHealthResult(
            ok=True,
            reachable=True,
            wp_detected=True,
            auth_ok=True,
            site_name="My Blog",
            site_url="https://example.com",
        )

    def test_success_returns_ok_true(self):
        """wp_health_check returns ok=True when service.health() succeeds."""
        from rex.openclaw.tools.wordpress_tool import wp_health_check

        mock_service = MagicMock()
        mock_service.health.return_value = self._make_healthy_result()

        with patch(
            "rex.openclaw.tools.wordpress_tool._get_wordpress_service",
            return_value=mock_service,
        ):
            result = wp_health_check("myblog")

        assert result["ok"] is True
        assert result["site_name"] == "My Blog"
        assert result["site_url"] == "https://example.com"
        assert result["wp_detected"] is True
        assert result["auth_ok"] is True
        assert result["error"] is None

    def test_site_id_forwarded_to_health(self):
        """wp_health_check passes site_id to service.health()."""
        from rex.openclaw.tools.wordpress_tool import wp_health_check

        mock_service = MagicMock()
        mock_service.health.return_value = self._make_healthy_result()

        with patch(
            "rex.openclaw.tools.wordpress_tool._get_wordpress_service",
            return_value=mock_service,
        ):
            wp_health_check("nasteeshirts")

        mock_service.health.assert_called_once_with("nasteeshirts")

    def test_auth_ok_none_when_no_credentials(self):
        """auth_ok=None when no credentials configured."""
        from rex.openclaw.tools.wordpress_tool import wp_health_check

        result_no_auth = WPHealthResult(
            ok=True,
            reachable=True,
            wp_detected=True,
            auth_ok=None,
            site_name="Public Blog",
            site_url="https://public.example.com",
        )
        mock_service = MagicMock()
        mock_service.health.return_value = result_no_auth

        with patch(
            "rex.openclaw.tools.wordpress_tool._get_wordpress_service",
            return_value=mock_service,
        ):
            result = wp_health_check("public")

        assert result["auth_ok"] is None

    def test_unreachable_site_returns_ok_false(self):
        """wp_health_check returns ok=False when site is unreachable."""
        from rex.openclaw.tools.wordpress_tool import wp_health_check

        mock_service = MagicMock()
        mock_service.health.return_value = WPHealthResult(
            ok=False,
            reachable=False,
            error="Connection refused",
        )

        with patch(
            "rex.openclaw.tools.wordpress_tool._get_wordpress_service",
            return_value=mock_service,
        ):
            result = wp_health_check("myblog")

        assert result["ok"] is False
        assert result["error"] == "Connection refused"

    def test_service_exception_returns_error_dict(self):
        """wp_health_check wraps service exceptions into an error dict."""
        from rex.openclaw.tools.wordpress_tool import wp_health_check

        mock_service = MagicMock()
        mock_service.health.side_effect = RuntimeError("Site config not found")

        with patch(
            "rex.openclaw.tools.wordpress_tool._get_wordpress_service",
            return_value=mock_service,
        ):
            result = wp_health_check("unknown_site")

        assert result["ok"] is False
        assert result["error"] is not None
        assert "Site config not found" in result["error"]

    def test_context_kwarg_accepted(self):
        """context kwarg is accepted without error."""
        from rex.openclaw.tools.wordpress_tool import wp_health_check

        mock_service = MagicMock()
        mock_service.health.return_value = self._make_healthy_result()

        with patch(
            "rex.openclaw.tools.wordpress_tool._get_wordpress_service",
            return_value=mock_service,
        ):
            result = wp_health_check("myblog", context={"user": "james"})

        assert result["ok"] is True

    def test_result_has_all_expected_keys(self):
        """Result dict always contains: ok, site_name, site_url, wp_detected, auth_ok, error."""
        from rex.openclaw.tools.wordpress_tool import wp_health_check

        mock_service = MagicMock()
        mock_service.health.return_value = self._make_healthy_result()

        with patch(
            "rex.openclaw.tools.wordpress_tool._get_wordpress_service",
            return_value=mock_service,
        ):
            result = wp_health_check("myblog")

        for key in ("ok", "site_name", "site_url", "wp_detected", "auth_ok", "error"):
            assert key in result, f"Key {key!r} missing from result"


# ---------------------------------------------------------------------------
# US-P5-011: Read-only assertion — no write tools
# ---------------------------------------------------------------------------


class TestWordpressReadOnly:
    """WordPress integration is read-only — no write tools registered."""

    def test_only_health_check_tool_registered(self):
        """register_wordpress_tools returns only wordpress_health_check."""
        from rex.openclaw.tool_bridge import ToolBridge

        result = ToolBridge().register_wordpress_tools()
        assert set(result.keys()) == {"wordpress_health_check"}

    def test_no_create_post_tool_exists(self):
        """No wp_create_post or similar write tool exists."""
        import rex.openclaw.tools.wordpress_tool as module

        assert not hasattr(module, "wp_create_post")
        assert not hasattr(module, "wp_update_post")
        assert not hasattr(module, "wp_delete_post")

    def test_health_check_is_read_only(self):
        """wp_health_check only calls service.health() — no write methods."""
        from rex.openclaw.tools.wordpress_tool import wp_health_check

        mock_service = MagicMock()
        mock_service.health.return_value = WPHealthResult(ok=True)

        with patch(
            "rex.openclaw.tools.wordpress_tool._get_wordpress_service",
            return_value=mock_service,
        ):
            wp_health_check("myblog")

        mock_service.health.assert_called_once()
        # No other methods were called (e.g. create, update, delete)
        called_methods = [c[0] for c in mock_service.method_calls if c[0] != "health"]
        assert called_methods == []


# ---------------------------------------------------------------------------
# ToolBridge.register_wordpress_tools
# ---------------------------------------------------------------------------


class TestRegisterWordpressTools:
    """ToolBridge.register_wordpress_tools() returns expected structure."""

    def test_returns_dict(self):
        from rex.openclaw.tool_bridge import ToolBridge

        result = ToolBridge().register_wordpress_tools()
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        from rex.openclaw.tool_bridge import ToolBridge

        result = ToolBridge().register_wordpress_tools()
        assert "wordpress_health_check" in result

    def test_calls_register_fn(self):
        """register_wordpress_tools calls the individual register function."""
        from rex.openclaw.tool_bridge import ToolBridge

        sentinel = object()

        with patch(
            "rex.openclaw.tool_bridge._register_wp_health_check", return_value=sentinel
        ) as mock_fn:
            result = ToolBridge().register_wordpress_tools()

        mock_fn.assert_called_once_with(agent=None)
        assert result["wordpress_health_check"] is sentinel

    def test_forwards_agent_arg(self):
        """agent kwarg is forwarded to the register function."""
        from rex.openclaw.tool_bridge import ToolBridge

        fake_agent = object()

        with patch(
            "rex.openclaw.tool_bridge._register_wp_health_check", return_value=None
        ) as mock_fn:
            ToolBridge().register_wordpress_tools(agent=fake_agent)

        mock_fn.assert_called_once_with(agent=fake_agent)

    def test_returns_none_values_without_openclaw(self):
        """Without openclaw, all values in the dict are None."""
        from rex.openclaw.tool_bridge import OPENCLAW_AVAILABLE, ToolBridge

        if not OPENCLAW_AVAILABLE:
            result = ToolBridge().register_wordpress_tools()
            assert all(v is None for v in result.values())
