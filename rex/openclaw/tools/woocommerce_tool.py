"""OpenClaw tool adapters — WooCommerce read and write operations.

Wraps Rex's :class:`~rex.woocommerce.service.WooCommerceService` to expose
WooCommerce operations for registration with OpenClaw's tool system.

**Read tools (LOW risk):**
- ``wc_list_orders``      — list orders (with optional status filter)
- ``wc_list_products``    — list products (with optional low-stock filter)

**Write tools (HIGH risk, approval-gated):**
- ``wc_set_order_status`` — update order status
- ``wc_create_coupon``    — create a coupon
- ``wc_disable_coupon``   — disable a coupon

Write tools consult Rex's write policy via
:func:`~rex.woocommerce.write_policy.check_wc_write_policy` before any
network call.  If policy requires approval a ``{"ok": False,
"approval_required": True, ...}`` dict is returned without contacting the API.

When the ``openclaw`` package is not installed, :func:`register` logs a
warning and returns ``None``.  All tool callables work independently of
OpenClaw.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool names
# ---------------------------------------------------------------------------

TOOL_LIST_ORDERS = "wc_list_orders"
TOOL_LIST_PRODUCTS = "wc_list_products"
TOOL_SET_ORDER_STATUS = "wc_set_order_status"
TOOL_CREATE_COUPON = "wc_create_coupon"
TOOL_DISABLE_COUPON = "wc_disable_coupon"

ALL_TOOL_NAMES = (
    TOOL_LIST_ORDERS,
    TOOL_LIST_PRODUCTS,
    TOOL_SET_ORDER_STATUS,
    TOOL_CREATE_COUPON,
    TOOL_DISABLE_COUPON,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_woocommerce_service() -> Any:
    """Return the WooCommerce service singleton (injectable in tests)."""
    from rex.woocommerce.service import get_woocommerce_service

    return get_woocommerce_service()


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------


def wc_list_orders(
    site_id: str,
    *,
    status: str | None = None,
    limit: int = 10,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """List WooCommerce orders from the specified site.

    Args:
        site_id: WooCommerce site identifier from ``woocommerce.sites[].id``.
        status: Optional order status filter (e.g. ``"pending"``, ``"completed"``).
        limit: Maximum number of orders to return (default 10, max 100).
        context: Optional ambient context dict (unused; reserved).

    Returns:
        Dict with ``ok`` (bool), ``orders`` (list), ``error`` (str | None).
    """
    try:
        service = _get_woocommerce_service()
        result = service.list_orders(site_id, status=status, limit=limit)
        return {"ok": result.ok, "orders": result.orders, "error": result.error}
    except Exception as exc:
        logger.warning("[WC tool] list_orders failed for site %r: %s", site_id, exc)
        return {"ok": False, "orders": [], "error": str(exc)}


def wc_list_products(
    site_id: str,
    *,
    limit: int = 10,
    low_stock: bool = False,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """List WooCommerce products from the specified site.

    Args:
        site_id: WooCommerce site identifier.
        limit: Maximum number of products to return (default 10, max 100).
        low_stock: When ``True``, filter to low-stock / out-of-stock products.
        context: Optional ambient context dict (unused; reserved).

    Returns:
        Dict with ``ok`` (bool), ``products`` (list), ``error`` (str | None).
    """
    try:
        service = _get_woocommerce_service()
        result = service.list_products(site_id, limit=limit, low_stock=low_stock)
        return {"ok": result.ok, "products": result.products, "error": result.error}
    except Exception as exc:
        logger.warning("[WC tool] list_products failed for site %r: %s", site_id, exc)
        return {"ok": False, "products": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Write tools (policy-gated)
# ---------------------------------------------------------------------------


def _check_write_policy(
    action: str,
    site_id: str,
    identifiers: dict[str, Any],
    params: dict[str, Any],
    step_description: str,
) -> dict[str, Any] | None:
    """Return a blocked-result dict if write policy blocks; None if allowed.

    Returns:
        ``None`` if the action is allowed to proceed.
        A result dict with ``approval_required=True`` or ``denied=True`` otherwise.
    """
    from rex.woocommerce.write_policy import check_wc_write_policy

    decision, approval = check_wc_write_policy(
        action=action,
        site_id=site_id,
        identifiers=identifiers,
        params=params,
        step_description=step_description,
    )

    if decision.denied:
        return {
            "ok": False,
            "denied": True,
            "approval_required": False,
            "error": f"Policy denied: {action}",
        }

    if decision.requires_approval:
        approval_id = approval.approval_id if approval else None
        return {
            "ok": False,
            "denied": False,
            "approval_required": True,
            "approval_id": approval_id,
            "error": f"Approval required before executing {action}",
        }

    return None  # auto-execute allowed


def wc_set_order_status(
    site_id: str,
    order_id: int,
    status: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update the status of a WooCommerce order (approval-gated, HIGH risk).

    Args:
        site_id: WooCommerce site identifier.
        order_id: The WooCommerce order ID.
        status: New order status (e.g. ``"completed"``, ``"cancelled"``).
        context: Optional ambient context dict (unused; reserved).

    Returns:
        Dict with ``ok``, ``data``, ``error``; or ``approval_required=True`` when blocked.
    """
    blocked = _check_write_policy(
        action=TOOL_SET_ORDER_STATUS,
        site_id=site_id,
        identifiers={"order_id": order_id, "status": status},
        params={"order_id": order_id, "status": status},
        step_description=f"Set order #{order_id} status to '{status}' on {site_id}",
    )
    if blocked is not None:
        return blocked

    try:
        service = _get_woocommerce_service()
        result = service.set_order_status(site_id, order_id, status=status)
        return {"ok": result.ok, "data": result.data, "error": result.error}
    except Exception as exc:
        logger.warning("[WC tool] set_order_status failed for site %r: %s", site_id, exc)
        return {"ok": False, "data": None, "error": str(exc)}


def wc_create_coupon(
    site_id: str,
    code: str,
    amount: str,
    discount_type: str,
    *,
    date_expires: str | None = None,
    usage_limit: int | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a WooCommerce coupon (approval-gated, HIGH risk).

    Args:
        site_id: WooCommerce site identifier.
        code: Coupon code.
        amount: Discount amount as string (e.g. ``"10"``).
        discount_type: ``"percent"``, ``"fixed_cart"``, or ``"fixed_product"``.
        date_expires: Optional expiry date in ``"YYYY-MM-DD"`` format.
        usage_limit: Optional maximum usage count.
        context: Optional ambient context dict (unused; reserved).

    Returns:
        Dict with ``ok``, ``data``, ``error``; or ``approval_required=True`` when blocked.
    """
    blocked = _check_write_policy(
        action=TOOL_CREATE_COUPON,
        site_id=site_id,
        identifiers={"code": code, "amount": amount, "discount_type": discount_type},
        params={
            "code": code,
            "amount": amount,
            "discount_type": discount_type,
            "date_expires": date_expires,
            "usage_limit": usage_limit,
        },
        step_description=f"Create coupon '{code}' ({discount_type} {amount}) on {site_id}",
    )
    if blocked is not None:
        return blocked

    try:
        service = _get_woocommerce_service()
        result = service.create_coupon(
            site_id,
            code=code,
            amount=amount,
            discount_type=discount_type,
            date_expires=date_expires,
            usage_limit=usage_limit,
        )
        return {"ok": result.ok, "data": result.data, "error": result.error}
    except Exception as exc:
        logger.warning("[WC tool] create_coupon failed for site %r: %s", site_id, exc)
        return {"ok": False, "data": None, "error": str(exc)}


def wc_disable_coupon(
    site_id: str,
    coupon_id: int,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Disable a WooCommerce coupon (approval-gated, HIGH risk).

    Args:
        site_id: WooCommerce site identifier.
        coupon_id: The WooCommerce coupon ID.
        context: Optional ambient context dict (unused; reserved).

    Returns:
        Dict with ``ok``, ``data``, ``error``; or ``approval_required=True`` when blocked.
    """
    blocked = _check_write_policy(
        action=TOOL_DISABLE_COUPON,
        site_id=site_id,
        identifiers={"coupon_id": coupon_id},
        params={"coupon_id": coupon_id},
        step_description=f"Disable coupon #{coupon_id} on {site_id}",
    )
    if blocked is not None:
        return blocked

    try:
        service = _get_woocommerce_service()
        result = service.disable_coupon(site_id, coupon_id)
        return {"ok": result.ok, "data": result.data, "error": result.error}
    except Exception as exc:
        logger.warning("[WC tool] disable_coupon failed for site %r: %s", site_id, exc)
        return {"ok": False, "data": None, "error": str(exc)}


