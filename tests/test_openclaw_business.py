"""Tests for rex.openclaw.tools.business_tool — US-P5-021 and US-P5-022.

US-P5-021 acceptance criteria:
  - register_all_business_tools() exists and returns a dict with all WC + WP tool names
  - ToolBridge.register_business_tools() exists and delegates correctly
  - OpenClaw-not-installed returns dict of Nones

US-P5-022 acceptance criteria (end-to-end business workflow):
  - list orders → check inventory → set order status (with policy gate) flows through tools
  - "list orders, then conditionally set status" workflow: read step succeeds,
    write step is approval-gated
  - Full workflow via ToolBridge: register_business_tools() covers all 6 tools
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import rex.woocommerce.service as _wc_service_module
import rex.wordpress.service as _wp_service_module
from rex.openclaw.tool_bridge import ToolBridge
from rex.openclaw.tools.business_tool import (
    ALL_BUSINESS_TOOLS,
    WOOCOMMERCE_TOOLS,
    WORDPRESS_TOOLS,
    register_all_business_tools,
)
from rex.openclaw.tools.woocommerce_tool import wc_list_orders, wc_set_order_status
from rex.openclaw.tools.wordpress_tool import wp_health_check

# ---------------------------------------------------------------------------
# US-P5-021: Module structure
# ---------------------------------------------------------------------------


class TestBusinessToolStructure:
    def test_all_business_tools_contains_wc_and_wp(self):
        assert set(WOOCOMMERCE_TOOLS).issubset(set(ALL_BUSINESS_TOOLS))
        assert set(WORDPRESS_TOOLS).issubset(set(ALL_BUSINESS_TOOLS))

    def test_all_business_tools_count(self):
        # 5 WC tools + 1 WP tool
        assert len(ALL_BUSINESS_TOOLS) == 6

    def test_register_all_returns_dict(self):
        result = register_all_business_tools()
        assert isinstance(result, dict)

    def test_register_all_keys_cover_wc_tools(self):
        result = register_all_business_tools()
        for name in WOOCOMMERCE_TOOLS:
            assert name in result

    def test_register_all_keys_cover_wp_tools(self):
        result = register_all_business_tools()
        for name in WORDPRESS_TOOLS:
            assert name in result

    def test_register_all_values_none_without_openclaw(self):
        result = register_all_business_tools()
        # openclaw not installed in test env; all handles should be None
        assert all(v is None for v in result.values())

    def test_tool_bridge_has_register_business_tools(self):
        bridge = ToolBridge()
        assert callable(bridge.register_business_tools)

    def test_tool_bridge_register_business_tools_returns_dict(self):
        bridge = ToolBridge()
        result = bridge.register_business_tools()
        assert isinstance(result, dict)
        for name in ALL_BUSINESS_TOOLS:
            assert name in result

    def test_tool_bridge_register_business_tools_delegates(self):
        bridge = ToolBridge()
        fake_handles = {name: MagicMock() for name in ALL_BUSINESS_TOOLS}

        with patch(
            "rex.openclaw.tool_bridge._register_business_tools",
            return_value=fake_handles,
        ) as mock_reg:
            result = bridge.register_business_tools(agent=None)

        mock_reg.assert_called_once_with(agent=None)
        assert result is fake_handles

    def test_tool_bridge_register_business_tools_passes_agent(self):
        bridge = ToolBridge()
        agent = MagicMock()

        with patch(
            "rex.openclaw.tool_bridge._register_business_tools",
            return_value={},
        ) as mock_reg:
            bridge.register_business_tools(agent=agent)

        mock_reg.assert_called_once_with(agent=agent)


# ---------------------------------------------------------------------------
# US-P5-022: End-to-end business workflow
# Scenario: list orders on "mystore" → if any pending, set first one to "processing"
# ---------------------------------------------------------------------------


class TestBusinessWorkflowEndToEnd:
    """Simulate a list-then-update business workflow through OpenClaw tools."""

    def _make_wc_service(self):
        from rex.woocommerce.service import WooCommerceService

        return MagicMock(spec=WooCommerceService)

    def _make_orders_result(self, orders):
        result = MagicMock()
        result.ok = True
        result.orders = orders
        result.error = None
        return result

    def _make_write_result(self, ok=True, error=None):
        result = MagicMock()
        result.ok = ok
        result.error = error
        return result

    def _make_wp_service(self):
        from rex.wordpress.service import WordPressService

        return MagicMock(spec=WordPressService)

    def test_list_orders_step(self):
        """Step 1: List orders returns ok=True with results."""
        svc = self._make_wc_service()
        svc.list_orders.return_value = self._make_orders_result([{"id": 101, "status": "pending"}])
        with patch.object(_wc_service_module, "_service", svc):
            result = wc_list_orders("mystore", status="pending")

        assert result["ok"] is True
        assert len(result["orders"]) == 1

    def test_set_order_status_step_requires_approval(self):
        """Step 2: Setting order status is approval-gated."""
        from rex.woocommerce.write_policy import PolicyDecision, WorkflowApproval

        decision = MagicMock(spec=PolicyDecision)
        decision.denied = False
        decision.requires_approval = True

        approval = MagicMock(spec=WorkflowApproval)
        approval.approval_id = "ap-biz-workflow-001"

        with patch(
            "rex.woocommerce.write_policy.check_wc_write_policy",
            return_value=(decision, approval),
        ):
            result = wc_set_order_status("mystore", "101", status="processing")

        assert result["ok"] is False
        assert result["approval_required"] is True
        assert result["approval_id"] == "ap-biz-workflow-001"

    def test_set_order_status_step_auto_allowed(self):
        """Step 2 (auto-allowed policy): order status update succeeds."""
        from rex.woocommerce.service import WooCommerceService
        from rex.woocommerce.write_policy import PolicyDecision

        decision = MagicMock(spec=PolicyDecision)
        decision.denied = False
        decision.requires_approval = False

        svc = MagicMock(spec=WooCommerceService)
        svc.set_order_status.return_value = self._make_write_result(ok=True)

        with patch(
            "rex.woocommerce.write_policy.check_wc_write_policy",
            return_value=(decision, None),
        ):
            with patch.object(_wc_service_module, "_service", svc):
                result = wc_set_order_status("mystore", "101", status="processing")

        assert result["ok"] is True
        svc.set_order_status.assert_called_once()

    def test_health_check_step(self):
        """Health check step succeeds."""
        from rex.wordpress.client import WPHealthResult
        from rex.wordpress.service import WordPressService

        svc = MagicMock(spec=WordPressService)
        health_result = MagicMock(spec=WPHealthResult)
        health_result.ok = True
        health_result.site_name = "My Store"
        health_result.site_url = "https://mystore.example.com"
        health_result.wp_detected = True
        health_result.auth_ok = True
        health_result.error = None
        svc.health.return_value = health_result

        with patch.object(_wp_service_module, "_service", svc):
            result = wp_health_check("mystore")

        assert result["ok"] is True
        assert result["site_name"] == "My Store"

    def test_full_workflow_list_check_update(self):
        """Full workflow: list_orders → health_check → set_order_status (auto-allowed)."""
        from rex.woocommerce.service import WooCommerceService
        from rex.woocommerce.write_policy import PolicyDecision
        from rex.wordpress.client import WPHealthResult
        from rex.wordpress.service import WordPressService

        # --- Setup mocks ---
        wc_svc = MagicMock(spec=WooCommerceService)
        wc_svc.list_orders.return_value = self._make_orders_result(
            [{"id": 201, "status": "pending"}]
        )
        wc_svc.set_order_status.return_value = self._make_write_result(ok=True)

        wp_svc = MagicMock(spec=WordPressService)
        wp_health = MagicMock(spec=WPHealthResult)
        wp_health.ok = True
        wp_health.site_name = "Nasteeshirts"
        wp_health.site_url = "https://nasteeshirts.example.com"
        wp_health.wp_detected = True
        wp_health.auth_ok = True
        wp_health.error = None
        wp_svc.health.return_value = wp_health

        policy_decision = MagicMock(spec=PolicyDecision)
        policy_decision.denied = False
        policy_decision.requires_approval = False

        # --- Execute workflow steps ---
        with patch.object(_wc_service_module, "_service", wc_svc):
            with patch.object(_wp_service_module, "_service", wp_svc):
                with patch(
                    "rex.woocommerce.write_policy.check_wc_write_policy",
                    return_value=(policy_decision, None),
                ):
                    # Step 1: list orders
                    step1 = wc_list_orders("nasteeshirts", status="pending")
                    # Step 2: check WordPress health
                    step2 = wp_health_check("nasteeshirts")
                    # Step 3: update first order
                    order_id = str(step1["orders"][0]["id"])
                    step3 = wc_set_order_status("nasteeshirts", order_id, status="processing")

        # --- Assert workflow outcomes ---
        assert step1["ok"] is True
        assert step2["ok"] is True
        assert step3["ok"] is True
        wc_svc.set_order_status.assert_called_once()

    def test_register_all_business_tools_delegates_to_wc_and_wp(self):
        """register_all_business_tools() calls both WC and WP register functions."""
        fake_wc = {name: MagicMock() for name in WOOCOMMERCE_TOOLS}
        fake_wp = None  # wordpress_tool.register() returns None

        with (
            patch(
                "rex.openclaw.tools.business_tool._register_wc_tools",
                return_value=fake_wc,
            ) as mock_wc,
            patch(
                "rex.openclaw.tools.business_tool._register_wp_tools",
                return_value=fake_wp,
            ) as mock_wp,
        ):
            result = register_all_business_tools(agent=None)

        mock_wc.assert_called_once_with(agent=None)
        mock_wp.assert_called_once_with(agent=None)
        # WC tool handles are present
        for name in WOOCOMMERCE_TOOLS:
            assert name in result
        # WP tool handle is added with the canonical name
        assert "wordpress_health_check" in result
