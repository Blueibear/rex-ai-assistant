"""OpenClaw business tools — US-P5-021.

Convenience module that lists all Rex business-domain tools (WordPress
and WooCommerce) in one place.

Rex's business-domain logic is entirely covered by the WooCommerce and
WordPress integrations already bridged in Phase 5 (US-P5-009, US-P5-013).
Tools are exposed via the ToolServer HTTP endpoint (US-007/US-008).

Typical usage::

    from rex.openclaw.tools.business_tool import ALL_BUSINESS_TOOLS

    print(ALL_BUSINESS_TOOLS)  # all WC + WP tool names
"""

from __future__ import annotations

import logging

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
