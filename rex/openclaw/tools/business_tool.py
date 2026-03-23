"""OpenClaw business tools — US-P5-021.

Convenience wrapper that registers all Rex business-domain tools (WordPress
and WooCommerce) as a single batch.

Rex's business-domain logic is entirely covered by the WooCommerce and
WordPress integrations already bridged in Phase 5 (US-P5-009, US-P5-013).
Workflows are created dynamically at runtime — no static business workflow
templates exist.  "Nasteeshirts" is the example site_id used in test
fixtures; production stores are configured in ``rex_config.json``.

This module provides :func:`register_all_business_tools` as a single call
point for registering all WC + WP tools with OpenClaw.

Typical usage::

    from rex.openclaw.tools.business_tool import register_all_business_tools

    handles = register_all_business_tools()
    # handles keys: all woocommerce + wordpress tool names
"""

from __future__ import annotations

import logging
from typing import Any

from rex.openclaw.tools.woocommerce_tool import register as _register_wc_tools
from rex.openclaw.tools.wordpress_tool import register as _register_wp_tools

logger = logging.getLogger(__name__)

# Tool names bundled by this module (informational — actual names come from
# the individual tool modules)
WOOCOMMERCE_TOOLS = (
    "wc_list_orders",
    "wc_list_products",
    "wc_set_order_status",
    "wc_create_coupon",
    "wc_disable_coupon",
)
WORDPRESS_TOOLS = ("wordpress_health_check",)
ALL_BUSINESS_TOOLS = WOOCOMMERCE_TOOLS + WORDPRESS_TOOLS


def register_all_business_tools(agent: Any = None) -> dict[str, Any]:
    """Register all business-domain tools (WooCommerce + WordPress) with OpenClaw.

    Delegates to:
    - :func:`rex.openclaw.tools.woocommerce_tool.register` (5 WC tools)
    - :func:`rex.openclaw.tools.wordpress_tool.register` (1 WP tool)

    Args:
        agent: Optional OpenClaw agent handle forwarded to each tool's
            :func:`register` call.

    Returns:
        A dict mapping every tool name to its registration handle (``None``
        when OpenClaw is not installed).
    """
    handles: dict[str, Any] = {}
    handles.update(_register_wc_tools(agent=agent))
    wp_handle = _register_wp_tools(agent=agent)
    if isinstance(wp_handle, dict):
        handles.update(wp_handle)
    else:
        handles["wordpress_health_check"] = wp_handle
    return handles
