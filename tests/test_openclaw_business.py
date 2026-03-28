"""Tests for rex.openclaw.tools.business_tool — US-P5-021 and US-P5-022.

US-P5-021 acceptance criteria:
  - ALL_BUSINESS_TOOLS contains all WC + WP tool names

US-P5-022 acceptance criteria (end-to-end business workflow):
  - list orders → check inventory → set order status (with policy gate) flows through tools
  - "list orders, then conditionally set status" workflow: read step succeeds,
    write step is approval-gated
  - Full workflow via tool callables: covers all 6 tools
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import rex.woocommerce.service as _wc_service_module
import rex.wordpress.service as _wp_service_module
from rex.openclaw.tools.business_tool import (
    ALL_BUSINESS_TOOLS,
    WOOCOMMERCE_TOOLS,
    WORDPRESS_TOOLS,
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
