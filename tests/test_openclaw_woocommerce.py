"""Tests for US-P5-012 through US-P5-015: WooCommerce OpenClaw integration.

US-P5-012 acceptance criteria:
  - WooCommerce public API documented (audit)

US-P5-013 acceptance criteria:
  - woocommerce_tool.py exists and is importable
  - All 5 tool names defined: wc_list_orders, wc_list_products, wc_set_order_status,
    wc_create_coupon, wc_disable_coupon
  - register() returns dict with all 5 tool names
  - ToolBridge.register_woocommerce_tools() delegates to register()

US-P5-014 acceptance criteria:
  - wc_list_orders() returns correct shape on success
  - wc_list_products() returns correct shape on success
  - Error conditions return ok=False

US-P5-015 acceptance criteria:
  - wc_set_order_status() blocked by policy (approval_required=True)
  - wc_create_coupon() blocked by policy
  - wc_disable_coupon() blocked by policy
  - Write tools execute when policy is auto-allowed
  - Write tools forward service exceptions
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rex.woocommerce.client import OrdersResult, ProductsResult, WriteResult


# ---------------------------------------------------------------------------
# US-P5-013: woocommerce_tool.py structure
# ---------------------------------------------------------------------------


class TestWoocommerceTool:
    def test_import(self):
        from rex.openclaw.tools import woocommerce_tool  # noqa: F401

    def test_all_tool_names_defined(self):
        from rex.openclaw.tools.woocommerce_tool import (
            TOOL_CREATE_COUPON,
            TOOL_DISABLE_COUPON,
            TOOL_LIST_ORDERS,
            TOOL_LIST_PRODUCTS,
            TOOL_SET_ORDER_STATUS,
        )

        assert TOOL_LIST_ORDERS == "wc_list_orders"
        assert TOOL_LIST_PRODUCTS == "wc_list_products"
        assert TOOL_SET_ORDER_STATUS == "wc_set_order_status"
        assert TOOL_CREATE_COUPON == "wc_create_coupon"
        assert TOOL_DISABLE_COUPON == "wc_disable_coupon"

    def test_register_returns_dict_with_all_keys(self):
        from rex.openclaw.tools.woocommerce_tool import ALL_TOOL_NAMES, register

        result = register()
        assert isinstance(result, dict)
        assert set(result.keys()) == set(ALL_TOOL_NAMES)

    def test_register_returns_none_values_without_openclaw(self):
        from rex.openclaw.tools.woocommerce_tool import OPENCLAW_AVAILABLE, register

        if not OPENCLAW_AVAILABLE:
            result = register()
            assert all(v is None for v in result.values())


# ---------------------------------------------------------------------------
# US-P5-014: Read tools
# ---------------------------------------------------------------------------


class TestWcListOrders:
    """wc_list_orders() correctly delegates to WooCommerceService.list_orders()."""

    def _make_service(self, orders=None, ok=True, error=None):
        svc = MagicMock()
        svc.list_orders.return_value = OrdersResult(
            ok=ok, orders=orders or [], error=error
        )
        return svc

    def test_returns_ok_true_on_success(self):
        from rex.openclaw.tools.woocommerce_tool import wc_list_orders

        svc = self._make_service(orders=[{"id": 1, "status": "pending"}])
        with patch(
            "rex.openclaw.tools.woocommerce_tool._get_woocommerce_service",
            return_value=svc,
        ):
            result = wc_list_orders("myshop")

        assert result["ok"] is True
        assert len(result["orders"]) == 1
        assert result["error"] is None

    def test_site_id_forwarded(self):
        from rex.openclaw.tools.woocommerce_tool import wc_list_orders

        svc = self._make_service()
        with patch(
            "rex.openclaw.tools.woocommerce_tool._get_woocommerce_service",
            return_value=svc,
        ):
            wc_list_orders("nasteeshirts")

        svc.list_orders.assert_called_once_with(
            "nasteeshirts", status=None, limit=10
        )

    def test_status_filter_forwarded(self):
        from rex.openclaw.tools.woocommerce_tool import wc_list_orders

        svc = self._make_service()
        with patch(
            "rex.openclaw.tools.woocommerce_tool._get_woocommerce_service",
            return_value=svc,
        ):
            wc_list_orders("myshop", status="pending", limit=5)

        svc.list_orders.assert_called_once_with("myshop", status="pending", limit=5)

    def test_service_exception_returns_error(self):
        from rex.openclaw.tools.woocommerce_tool import wc_list_orders

        svc = MagicMock()
        svc.list_orders.side_effect = RuntimeError("Connection failed")

        with patch(
            "rex.openclaw.tools.woocommerce_tool._get_woocommerce_service",
            return_value=svc,
        ):
            result = wc_list_orders("myshop")

        assert result["ok"] is False
        assert "Connection failed" in result["error"]
        assert result["orders"] == []

    def test_result_has_expected_keys(self):
        from rex.openclaw.tools.woocommerce_tool import wc_list_orders

        svc = self._make_service()
        with patch(
            "rex.openclaw.tools.woocommerce_tool._get_woocommerce_service",
            return_value=svc,
        ):
            result = wc_list_orders("myshop")

        for key in ("ok", "orders", "error"):
            assert key in result


class TestWcListProducts:
    """wc_list_products() correctly delegates to WooCommerceService.list_products()."""

    def _make_service(self, products=None, ok=True):
        svc = MagicMock()
        svc.list_products.return_value = ProductsResult(
            ok=ok, products=products or []
        )
        return svc

    def test_returns_ok_true_on_success(self):
        from rex.openclaw.tools.woocommerce_tool import wc_list_products

        svc = self._make_service(products=[{"id": 5, "name": "T-Shirt"}])
        with patch(
            "rex.openclaw.tools.woocommerce_tool._get_woocommerce_service",
            return_value=svc,
        ):
            result = wc_list_products("nasteeshirts")

        assert result["ok"] is True
        assert len(result["products"]) == 1

    def test_low_stock_flag_forwarded(self):
        from rex.openclaw.tools.woocommerce_tool import wc_list_products

        svc = self._make_service()
        with patch(
            "rex.openclaw.tools.woocommerce_tool._get_woocommerce_service",
            return_value=svc,
        ):
            wc_list_products("myshop", low_stock=True)

        svc.list_products.assert_called_once_with("myshop", limit=10, low_stock=True)

    def test_service_exception_returns_error(self):
        from rex.openclaw.tools.woocommerce_tool import wc_list_products

        svc = MagicMock()
        svc.list_products.side_effect = RuntimeError("Timeout")

        with patch(
            "rex.openclaw.tools.woocommerce_tool._get_woocommerce_service",
            return_value=svc,
        ):
            result = wc_list_products("myshop")

        assert result["ok"] is False
        assert result["products"] == []


# ---------------------------------------------------------------------------
# US-P5-015: Write tools — policy gate
# ---------------------------------------------------------------------------


def _make_approval_required_policy():
    """Return a mock (decision, approval) for 'requires_approval' outcome."""
    from rex.policy import PolicyDecision

    decision = PolicyDecision(
        allowed=True,
        requires_approval=True,
        denied=False,
        reason="Tool requires approval",
    )
    approval = MagicMock()
    approval.approval_id = "approval-wc-abc123"
    return decision, approval


def _make_denied_policy():
    from rex.policy import PolicyDecision

    decision = PolicyDecision(
        allowed=False,
        requires_approval=False,
        denied=True,
        reason="Tool is denied by policy",
    )
    return decision, None


def _make_auto_policy():
    from rex.policy import PolicyDecision

    decision = PolicyDecision(
        allowed=True,
        requires_approval=False,
        denied=False,
        reason="Auto-execute allowed",
    )
    return decision, None


class TestWcSetOrderStatusPolicy:
    def test_blocked_by_policy_returns_approval_required(self):
        from rex.openclaw.tools.woocommerce_tool import wc_set_order_status

        with patch(
            "rex.woocommerce.write_policy.check_wc_write_policy",
            return_value=_make_approval_required_policy(),
        ):
            result = wc_set_order_status("myshop", 42, "completed")

        assert result["ok"] is False
        assert result["approval_required"] is True
        assert result["approval_id"] == "approval-wc-abc123"

    def test_denied_policy_returns_denied(self):
        from rex.openclaw.tools.woocommerce_tool import wc_set_order_status

        with patch(
            "rex.woocommerce.write_policy.check_wc_write_policy",
            return_value=_make_denied_policy(),
        ):
            result = wc_set_order_status("myshop", 42, "completed")

        assert result["ok"] is False
        assert result["denied"] is True

    def test_auto_allowed_executes(self):
        from rex.openclaw.tools.woocommerce_tool import wc_set_order_status

        mock_svc = MagicMock()
        mock_svc.set_order_status.return_value = WriteResult(ok=True, data={"id": 42})

        with patch(
            "rex.woocommerce.write_policy.check_wc_write_policy",
            return_value=_make_auto_policy(),
        ):
            with patch(
                "rex.openclaw.tools.woocommerce_tool._get_woocommerce_service",
                return_value=mock_svc,
            ):
                result = wc_set_order_status("myshop", 42, "completed")

        assert result["ok"] is True
        mock_svc.set_order_status.assert_called_once_with("myshop", 42, status="completed")

    def test_service_exception_after_policy_returns_error(self):
        from rex.openclaw.tools.woocommerce_tool import wc_set_order_status

        mock_svc = MagicMock()
        mock_svc.set_order_status.side_effect = RuntimeError("API failure")

        with patch(
            "rex.woocommerce.write_policy.check_wc_write_policy",
            return_value=_make_auto_policy(),
        ):
            with patch(
                "rex.openclaw.tools.woocommerce_tool._get_woocommerce_service",
                return_value=mock_svc,
            ):
                result = wc_set_order_status("myshop", 42, "completed")

        assert result["ok"] is False
        assert "API failure" in result["error"]


class TestWcCreateCouponPolicy:
    def test_blocked_by_policy(self):
        from rex.openclaw.tools.woocommerce_tool import wc_create_coupon

        with patch(
            "rex.woocommerce.write_policy.check_wc_write_policy",
            return_value=_make_approval_required_policy(),
        ):
            result = wc_create_coupon("myshop", "SUMMER20", "20", "percent")

        assert result["ok"] is False
        assert result["approval_required"] is True

    def test_auto_allowed_executes(self):
        from rex.openclaw.tools.woocommerce_tool import wc_create_coupon

        mock_svc = MagicMock()
        mock_svc.create_coupon.return_value = WriteResult(ok=True, data={"id": 99})

        with patch(
            "rex.woocommerce.write_policy.check_wc_write_policy",
            return_value=_make_auto_policy(),
        ):
            with patch(
                "rex.openclaw.tools.woocommerce_tool._get_woocommerce_service",
                return_value=mock_svc,
            ):
                result = wc_create_coupon("myshop", "SUMMER20", "20", "percent")

        assert result["ok"] is True
        mock_svc.create_coupon.assert_called_once()


class TestWcDisableCouponPolicy:
    def test_blocked_by_policy(self):
        from rex.openclaw.tools.woocommerce_tool import wc_disable_coupon

        with patch(
            "rex.woocommerce.write_policy.check_wc_write_policy",
            return_value=_make_approval_required_policy(),
        ):
            result = wc_disable_coupon("myshop", 77)

        assert result["ok"] is False
        assert result["approval_required"] is True

    def test_auto_allowed_executes(self):
        from rex.openclaw.tools.woocommerce_tool import wc_disable_coupon

        mock_svc = MagicMock()
        mock_svc.disable_coupon.return_value = WriteResult(ok=True)

        with patch(
            "rex.woocommerce.write_policy.check_wc_write_policy",
            return_value=_make_auto_policy(),
        ):
            with patch(
                "rex.openclaw.tools.woocommerce_tool._get_woocommerce_service",
                return_value=mock_svc,
            ):
                result = wc_disable_coupon("myshop", 77)

        assert result["ok"] is True
        mock_svc.disable_coupon.assert_called_once_with("myshop", 77)


# ---------------------------------------------------------------------------
# ToolBridge.register_woocommerce_tools
# ---------------------------------------------------------------------------


class TestRegisterWoocommerceTools:
    def test_returns_dict(self):
        from rex.openclaw.tool_bridge import ToolBridge

        result = ToolBridge().register_woocommerce_tools()
        assert isinstance(result, dict)

    def test_has_all_expected_keys(self):
        from rex.openclaw.tool_bridge import ToolBridge
        from rex.openclaw.tools.woocommerce_tool import ALL_TOOL_NAMES

        result = ToolBridge().register_woocommerce_tools()
        assert set(result.keys()) == set(ALL_TOOL_NAMES)

    def test_calls_register_fn(self):
        from rex.openclaw.tool_bridge import ToolBridge

        sentinel = {"wc_list_orders": None}
        with patch(
            "rex.openclaw.tool_bridge._register_wc_tools", return_value=sentinel
        ) as mock_fn:
            result = ToolBridge().register_woocommerce_tools()

        mock_fn.assert_called_once_with(agent=None)
        assert result is sentinel

    def test_forwards_agent_arg(self):
        from rex.openclaw.tool_bridge import ToolBridge

        fake_agent = object()
        with patch(
            "rex.openclaw.tool_bridge._register_wc_tools", return_value={}
        ) as mock_fn:
            ToolBridge().register_woocommerce_tools(agent=fake_agent)

        mock_fn.assert_called_once_with(agent=fake_agent)
