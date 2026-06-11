"""WordPress and WooCommerce commands for the Rex CLI.

Extracted verbatim from ``rex/cli.py`` (US-REM-027). Handler behavior,
argument definitions, help text, defaults, and exit codes are unchanged.
"""

from __future__ import annotations

import argparse
import sys


def _cli():
    """Return the ``rex.cli`` module at call time.

    ``rex.cli`` is the single patch point for service getters and command
    handlers (tests monkeypatch ``rex.cli.<name>``). Resolving through the
    module at call time preserves that behavior without creating an import
    cycle at module load time.
    """
    from rex import cli

    return cli


def cmd_wp(args: argparse.Namespace) -> int:
    """WordPress site monitoring (read-only)."""
    subcommand = args.wp_command

    if subcommand == "health":
        from rex.wordpress.service import (
            WordPressMissingCredentialError,
            WordPressSiteDisabledError,
            WordPressSiteNotFoundError,
            get_wordpress_service,
        )

        site_id = args.site
        service = get_wordpress_service()

        try:
            result = service.health(site_id)
        except WordPressSiteNotFoundError as e:
            print(f"Error: {e}")
            return 1
        except WordPressSiteDisabledError as e:
            print(f"Error: {e}")
            return 1
        except WordPressMissingCredentialError as e:
            print(f"Error: {e}")
            return 1

        print(f"WordPress Health: {site_id}")
        print("=" * 60)
        print(f"  Reachable    : {'Yes' if result.reachable else 'No'}")
        print(f"  WP detected  : {'Yes' if result.wp_detected else 'No'}")
        if result.site_name:
            print(f"  Site name    : {result.site_name}")
        if result.site_url:
            print(f"  Site URL     : {result.site_url}")
        if result.auth_ok is not None:
            print(f"  Auth check   : {'OK' if result.auth_ok else 'FAILED'}")
        if result.error:
            print(f"  Error        : {result.error}")
        print()
        status = "OK" if result.ok else "FAILED"
        print(f"Overall status: {status}")
        return 0 if result.ok else 1

    print("Unknown wp subcommand. Use 'rex wp --help'")
    return 1


def cmd_wc(args: argparse.Namespace) -> int:
    """WooCommerce monitoring + approval-gated write actions."""
    subcommand = args.wc_command

    if subcommand == "orders":
        wc_orders_cmd = getattr(args, "wc_orders_command", None)
        if wc_orders_cmd == "list":
            from rex.woocommerce.service import (
                WooCommerceMissingCredentialError,
                WooCommerceSiteDisabledError,
                WooCommerceSiteNotFoundError,
                get_woocommerce_service,
            )

            site_id = args.site
            status_filter = getattr(args, "status", None)
            limit = getattr(args, "limit", 10)
            service = get_woocommerce_service()

            try:
                result = service.list_orders(site_id, status=status_filter, limit=limit)
            except WooCommerceSiteNotFoundError as e:
                print(f"Error: {e}")
                return 1
            except WooCommerceSiteDisabledError as e:
                print(f"Error: {e}")
                return 1
            except WooCommerceMissingCredentialError as e:
                print(f"Error: {e}")
                return 1

            if not result.ok:
                print(f"Error: {result.error}")
                return 1

            status_label = f" [{status_filter}]" if status_filter else ""
            print(f"WooCommerce Orders: {site_id}{status_label}")
            print("=" * 60)

            if not result.orders:
                print("No orders found.")
                return 0

            for order in result.orders:
                order_id = order.get("id", "?")
                order_status = order.get("status", "unknown")
                total = order.get("total", "0.00")
                currency = order.get("currency", "")
                date_created = str(order.get("date_created", ""))[:10]
                billing = order.get("billing", {})
                customer = ""
                if billing:
                    first = billing.get("first_name", "")
                    last = billing.get("last_name", "")
                    customer = f" | {first} {last}".rstrip()
                print(
                    f"  #{order_id}  [{order_status}]  {currency} {total}"
                    f"  {date_created}{customer}"
                )

            print()
            print(f"Total: {len(result.orders)} order(s)")
            return 0

        if wc_orders_cmd == "set-status":
            return _cmd_wc_order_set_status(args)

        print("Unknown wc orders subcommand. Use 'rex wc orders --help'")
        return 1

    if subcommand == "products":
        wc_products_cmd = getattr(args, "wc_products_command", None)
        if wc_products_cmd == "list":
            from rex.woocommerce.service import (
                WooCommerceMissingCredentialError,
                WooCommerceSiteDisabledError,
                WooCommerceSiteNotFoundError,
                get_woocommerce_service,
            )

            site_id = args.site
            limit = getattr(args, "limit", 10)
            low_stock = getattr(args, "low_stock", False)
            service = get_woocommerce_service()

            try:
                result = service.list_products(site_id, limit=limit, low_stock=low_stock)  # type: ignore[assignment]
            except WooCommerceSiteNotFoundError as e:
                print(f"Error: {e}")
                return 1
            except WooCommerceSiteDisabledError as e:
                print(f"Error: {e}")
                return 1
            except WooCommerceMissingCredentialError as e:
                print(f"Error: {e}")
                return 1

            if not result.ok:
                print(f"Error: {result.error}")
                return 1

            stock_label = " [low-stock]" if low_stock else ""
            print(f"WooCommerce Products: {site_id}{stock_label}")
            print("=" * 60)

            if not result.products:  # type: ignore[attr-defined]
                no_msg = "No low-stock products found." if low_stock else "No products found."
                print(no_msg)
                return 0

            for product in result.products:  # type: ignore[attr-defined]
                product_id = product.get("id", "?")
                name = product.get("name", "Unknown")
                stock_status = product.get("stock_status", "")
                qty = product.get("stock_quantity")
                manage = product.get("manage_stock", False)
                stock_info = ""
                if manage and qty is not None:
                    stock_info = f"  stock={qty}"
                elif stock_status:
                    stock_info = f"  [{stock_status}]"
                price = product.get("price", "")
                price_str = f"  ${price}" if price else ""
                print(f"  #{product_id}  {name}{price_str}{stock_info}")

            print()
            print(f"Total: {len(result.products)} product(s)")  # type: ignore[attr-defined]
            return 0

        print("Unknown wc products subcommand. Use 'rex wc products --help'")
        return 1

    if subcommand == "coupons":
        wc_coupons_cmd = getattr(args, "wc_coupons_command", None)

        if wc_coupons_cmd == "create":
            return _cmd_wc_coupon_create(args)

        if wc_coupons_cmd == "disable":
            return _cmd_wc_coupon_disable(args)

        print("Unknown wc coupons subcommand. Use 'rex wc coupons --help'")
        return 1

    print("Unknown wc subcommand. Use 'rex wc --help'")
    return 1


# ---------------------------------------------------------------------------
# WooCommerce write action helpers (Cycle 6.3)
# ---------------------------------------------------------------------------

_WC_WRITE_HELP = (
    "Write actions require policy approval before they can execute.\n"
    "  Step 1: Run the command to create a pending approval record.\n"
    "  Step 2: Approve it:  rex approvals --approve <id>\n"
    "  Step 3: Re-run with --yes to execute."
)


def _resolve_wc_initiated_by(args: argparse.Namespace) -> str | None:
    """Resolve the active user identity for approval records (best-effort)."""
    try:
        from rex.identity import resolve_active_user

        return resolve_active_user(getattr(args, "user", None))
    except Exception:  # noqa: BLE001
        return None


def _cmd_wc_order_set_status(args: argparse.Namespace) -> int:
    """Handle ``rex wc orders set-status``."""

    from rex.woocommerce.service import (
        WooCommerceMissingCredentialError,
        WooCommerceSiteDisabledError,
        WooCommerceSiteNotFoundError,
        get_woocommerce_service,
    )
    from rex.woocommerce.write_policy import (
        WC_ORDER_SET_STATUS_TOOL,
        check_wc_write_policy,
    )

    site_id: str = args.site
    order_id: int = args.order_id
    new_status: str = args.status
    note: str | None = getattr(args, "note", None)
    yes: bool = getattr(args, "yes", False)

    initiated_by = _resolve_wc_initiated_by(args)

    # ------------------------------------------------------------------
    # Policy + approvals gate (evaluated BEFORE the --yes guard).
    # ------------------------------------------------------------------
    identifiers = {"order_id": order_id, "status": new_status}
    params = {"order_id": order_id, "status": new_status}
    if note:
        params["note"] = note

    policy_decision, approval = check_wc_write_policy(
        action=WC_ORDER_SET_STATUS_TOOL,
        site_id=site_id,
        identifiers=identifiers,
        params=params,
        step_description=f"Update order #{order_id} status to {new_status!r} on site {site_id!r}",
        initiated_by=initiated_by,
    )

    if policy_decision.denied:
        print(f"Error: WooCommerce write action denied by policy: {policy_decision.reason}")
        return 1

    if policy_decision.requires_approval:
        if approval is None or approval.status != "approved":
            if approval is not None:
                print("Approval required before this WooCommerce write action can proceed.")
                print()
                print(f"  Approval ID : {approval.approval_id}")
                print(f"  Site        : {site_id}")
                print(f"  Action      : set order #{order_id} status → {new_status!r}")
                if note:
                    print(f"  Note        : {note!r}")
                if initiated_by:
                    print(f"  Requested by: {initiated_by}")
                print()
                print(f"  To approve : rex approvals --approve {approval.approval_id}")
                print(f"  To deny    : rex approvals --deny {approval.approval_id}")
                print()
                print("After approving, re-run this command with --yes to execute.")
                print()
                print(_WC_WRITE_HELP)
            else:
                print("Error: Approval required but could not create approval record.")
            return 1

    # ------------------------------------------------------------------
    # Second-layer --yes confirmation guard.
    # ------------------------------------------------------------------
    if not yes:
        print("Refusing to update order status without explicit confirmation.")
        print(
            f"Re-run with '--yes' to update order #{order_id} status to {new_status!r} "
            f"on site {site_id!r}."
        )
        return 1

    # Execute
    service = get_woocommerce_service()
    try:
        result = service.set_order_status(site_id, order_id, status=new_status)
    except WooCommerceSiteNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except WooCommerceSiteDisabledError as e:
        print(f"Error: {e}")
        return 1
    except WooCommerceMissingCredentialError as e:
        print(f"Error: {e}")
        return 1

    if not result.ok:
        print(f"Error: {result.error}")
        return 1

    updated_id = (result.data or {}).get("id", order_id)
    updated_status = (result.data or {}).get("status", new_status)
    print(f"Order #{updated_id} status updated to {updated_status!r}.")

    if note:
        note_result = service.add_order_note(site_id, order_id, note=note, customer_note=False)
        if not note_result.ok:
            print(f"Warning: Order note could not be added: {note_result.error}", file=sys.stderr)
        else:
            print("Order note added.")

    return 0


def _cmd_wc_coupon_create(args: argparse.Namespace) -> int:
    """Handle ``rex wc coupons create``."""
    from rex.woocommerce.service import (
        WooCommerceMissingCredentialError,
        WooCommerceSiteDisabledError,
        WooCommerceSiteNotFoundError,
        get_woocommerce_service,
    )
    from rex.woocommerce.write_policy import (
        WC_COUPON_CREATE_TOOL,
        check_wc_write_policy,
    )

    site_id: str = args.site
    code: str = args.code.strip()
    amount_str: str = args.amount
    discount_type: str = args.type
    expires: str | None = getattr(args, "expires", None)
    usage_limit: int | None = getattr(args, "usage_limit", None)
    yes: bool = getattr(args, "yes", False)

    # Local input validation
    if not code:
        print("Error: --code must be a non-empty string.")
        return 1

    try:
        amount_val = float(amount_str)
        if amount_val <= 0:
            raise ValueError("amount must be positive")
    except ValueError:
        print(f"Error: --amount must be a positive number, got: {amount_str!r}")
        return 1

    allowed_types = {"percent", "fixed_cart", "fixed_product"}
    if discount_type not in allowed_types:
        print(f"Error: --type must be one of {sorted(allowed_types)}, got: {discount_type!r}")
        return 1

    if expires is not None:
        try:
            from datetime import datetime

            datetime.strptime(expires, "%Y-%m-%d")
        except ValueError:
            print(f"Error: --expires must be in YYYY-MM-DD format, got: {expires!r}")
            return 1

    initiated_by = _resolve_wc_initiated_by(args)

    # ------------------------------------------------------------------
    # Policy + approvals gate.
    # ------------------------------------------------------------------
    identifiers = {"code": code, "amount": amount_str, "discount_type": discount_type}
    params: dict = {
        "code": code,
        "amount": amount_str,
        "discount_type": discount_type,
    }
    if expires:
        params["date_expires"] = expires
    if usage_limit is not None:
        params["usage_limit"] = usage_limit

    policy_decision, approval = check_wc_write_policy(
        action=WC_COUPON_CREATE_TOOL,
        site_id=site_id,
        identifiers=identifiers,
        params=params,
        step_description=(
            f"Create coupon {code!r} ({discount_type} {amount_str}) on site {site_id!r}"
        ),
        initiated_by=initiated_by,
    )

    if policy_decision.denied:
        print(f"Error: WooCommerce write action denied by policy: {policy_decision.reason}")
        return 1

    if policy_decision.requires_approval:
        if approval is None or approval.status != "approved":
            if approval is not None:
                print("Approval required before this WooCommerce write action can proceed.")
                print()
                print(f"  Approval ID  : {approval.approval_id}")
                print(f"  Site         : {site_id}")
                print(f"  Action       : create coupon {code!r}")
                print(f"  Type / Amount: {discount_type} / {amount_str}")
                if expires:
                    print(f"  Expires      : {expires}")
                if usage_limit is not None:
                    print(f"  Usage limit  : {usage_limit}")
                if initiated_by:
                    print(f"  Requested by : {initiated_by}")
                print()
                print(f"  To approve : rex approvals --approve {approval.approval_id}")
                print(f"  To deny    : rex approvals --deny {approval.approval_id}")
                print()
                print("After approving, re-run this command with --yes to execute.")
                print()
                print(_WC_WRITE_HELP)
            else:
                print("Error: Approval required but could not create approval record.")
            return 1

    # ------------------------------------------------------------------
    # Second-layer --yes confirmation guard.
    # ------------------------------------------------------------------
    if not yes:
        print("Refusing to create coupon without explicit confirmation.")
        print(f"Re-run with '--yes' to create coupon {code!r} on site {site_id!r}.")
        return 1

    # Execute
    service = get_woocommerce_service()
    try:
        result = service.create_coupon(
            site_id,
            code=code,
            amount=amount_str,
            discount_type=discount_type,
            date_expires=expires,
            usage_limit=usage_limit,
        )
    except WooCommerceSiteNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except WooCommerceSiteDisabledError as e:
        print(f"Error: {e}")
        return 1
    except WooCommerceMissingCredentialError as e:
        print(f"Error: {e}")
        return 1

    if not result.ok:
        print(f"Error: {result.error}")
        return 1

    created_id = (result.data or {}).get("id", "?")
    created_code = (result.data or {}).get("code", code)
    print(f"Coupon created: #{created_id} code={created_code!r}")
    return 0


def _cmd_wc_coupon_disable(args: argparse.Namespace) -> int:
    """Handle ``rex wc coupons disable``."""
    from rex.woocommerce.service import (
        WooCommerceMissingCredentialError,
        WooCommerceSiteDisabledError,
        WooCommerceSiteNotFoundError,
        get_woocommerce_service,
    )
    from rex.woocommerce.write_policy import (
        WC_COUPON_DISABLE_TOOL,
        check_wc_write_policy,
    )

    site_id: str = args.site
    coupon_id: int = args.coupon_id
    yes: bool = getattr(args, "yes", False)

    # Local input validation
    if coupon_id <= 0:
        print(f"Error: --coupon-id must be a positive integer, got: {coupon_id}")
        return 1

    initiated_by = _resolve_wc_initiated_by(args)

    # ------------------------------------------------------------------
    # Policy + approvals gate.
    # ------------------------------------------------------------------
    identifiers = {"coupon_id": coupon_id}
    params = {"coupon_id": coupon_id}

    policy_decision, approval = check_wc_write_policy(
        action=WC_COUPON_DISABLE_TOOL,
        site_id=site_id,
        identifiers=identifiers,
        params=params,
        step_description=f"Disable coupon #{coupon_id} on site {site_id!r}",
        initiated_by=initiated_by,
    )

    if policy_decision.denied:
        print(f"Error: WooCommerce write action denied by policy: {policy_decision.reason}")
        return 1

    if policy_decision.requires_approval:
        if approval is None or approval.status != "approved":
            if approval is not None:
                print("Approval required before this WooCommerce write action can proceed.")
                print()
                print(f"  Approval ID : {approval.approval_id}")
                print(f"  Site        : {site_id}")
                print(f"  Action      : disable coupon #{coupon_id}")
                if initiated_by:
                    print(f"  Requested by: {initiated_by}")
                print()
                print(f"  To approve : rex approvals --approve {approval.approval_id}")
                print(f"  To deny    : rex approvals --deny {approval.approval_id}")
                print()
                print("After approving, re-run this command with --yes to execute.")
                print()
                print(_WC_WRITE_HELP)
            else:
                print("Error: Approval required but could not create approval record.")
            return 1

    # ------------------------------------------------------------------
    # Second-layer --yes confirmation guard.
    # ------------------------------------------------------------------
    if not yes:
        print("Refusing to disable coupon without explicit confirmation.")
        print(f"Re-run with '--yes' to disable coupon #{coupon_id} on site {site_id!r}.")
        return 1

    # Execute
    service = get_woocommerce_service()
    try:
        result = service.disable_coupon(site_id, coupon_id)
    except WooCommerceSiteNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except WooCommerceSiteDisabledError as e:
        print(f"Error: {e}")
        return 1
    except WooCommerceMissingCredentialError as e:
        print(f"Error: {e}")
        return 1

    if not result.ok:
        print(f"Error: {result.error}")
        return 1

    updated_id = (result.data or {}).get("id", coupon_id)
    updated_status = (result.data or {}).get("status", "draft")
    print(f"Coupon #{updated_id} disabled (status={updated_status!r}).")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register this domain's subcommands on the top-level subparsers."""
    # wp (WordPress read-only monitoring)
    wp_parser = subparsers.add_parser(
        "wp",
        help="WordPress site monitoring (read-only)",
        description=(
            "Monitor WordPress sites via the WP REST API. "
            "Requires a site entry in wordpress.sites[] in rex_config.json. "
            "Credentials are looked up via CredentialManager — never stored in config."
        ),
    )
    wp_subparsers = wp_parser.add_subparsers(
        title="wp commands",
        dest="wp_command",
        metavar="COMMAND",
    )

    wp_health = wp_subparsers.add_parser(
        "health",
        help="Check that a WordPress site is reachable and looks like WP",
        description=(
            "Calls GET /wp-json to verify reachability and WP detection. "
            "If auth is configured, also calls GET /wp-json/wp/v2/users/me."
        ),
    )
    wp_health.add_argument(
        "--site",
        type=str,
        required=True,
        help="WordPress site ID from wordpress.sites[] in config",
    )
    wp_health.set_defaults(func=_cli().cmd_wp, wp_command="health")

    wp_parser.set_defaults(func=_cli().cmd_wp, wp_command="health")

    # wc (WooCommerce monitoring + approval-gated writes)
    wc_parser = subparsers.add_parser(
        "wc",
        help="WooCommerce monitoring + approval-gated write actions",
        description=(
            "Monitor WooCommerce stores and run approval-gated write actions "
            "via the WC REST API v3. "
            "Requires a site entry in woocommerce.sites[] in rex_config.json. "
            "Consumer key and secret are looked up via CredentialManager — never stored in config."
        ),
    )
    wc_subparsers = wc_parser.add_subparsers(
        title="wc commands",
        dest="wc_command",
        metavar="COMMAND",
    )

    # wc orders
    wc_orders_parser = wc_subparsers.add_parser(
        "orders",
        help="Manage WooCommerce orders",
        description="WooCommerce orders subcommands.",
    )
    wc_orders_subparsers = wc_orders_parser.add_subparsers(
        title="orders commands",
        dest="wc_orders_command",
        metavar="COMMAND",
    )

    wc_orders_list = wc_orders_subparsers.add_parser(
        "list",
        help="List WooCommerce orders",
        description="Fetch orders from a WooCommerce site via the REST API v3.",
    )
    wc_orders_list.add_argument(
        "--site",
        type=str,
        required=True,
        help="WooCommerce site ID from woocommerce.sites[] in config",
    )
    wc_orders_list.add_argument(
        "--status",
        type=str,
        default=None,
        help=(
            "Filter orders by status "
            "(e.g. pending, processing, completed, on-hold, cancelled, refunded, failed)"
        ),
    )
    wc_orders_list.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of orders to return (default: 10, max: 100)",
    )
    wc_orders_list.set_defaults(func=_cli().cmd_wc, wc_command="orders", wc_orders_command="list")

    wc_orders_parser.set_defaults(func=_cli().cmd_wc, wc_command="orders", wc_orders_command="list")

    wc_orders_set_status = wc_orders_subparsers.add_parser(
        "set-status",
        help="Update a WooCommerce order status (requires approval + --yes)",
        description=(
            "Update the status of a WooCommerce order via the REST API v3.\n\n"
            "This is a write action gated by policy approval:\n"
            "  Step 1: Run without --yes to create a pending approval.\n"
            "  Step 2: Approve: rex approvals --approve <id>\n"
            "  Step 3: Re-run with --yes to execute."
        ),
    )
    wc_orders_set_status.add_argument(
        "--site",
        type=str,
        required=True,
        help="WooCommerce site ID from woocommerce.sites[] in config",
    )
    wc_orders_set_status.add_argument(
        "--order-id",
        dest="order_id",
        type=int,
        required=True,
        help="WooCommerce order ID",
    )
    wc_orders_set_status.add_argument(
        "--status",
        type=str,
        required=True,
        help=(
            "New order status "
            "(e.g. pending, processing, on-hold, completed, cancelled, refunded, failed)"
        ),
    )
    wc_orders_set_status.add_argument(
        "--note",
        type=str,
        default=None,
        help="Optional internal note to add to the order after the status change",
    )
    wc_orders_set_status.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Confirm execution (required after approval is granted)",
    )
    wc_orders_set_status.add_argument(
        "--user",
        type=str,
        default=None,
        help="Override active user identity for the approval record",
    )
    wc_orders_set_status.set_defaults(
        func=cmd_wc, wc_command="orders", wc_orders_command="set-status"
    )

    # wc products
    wc_products_parser = wc_subparsers.add_parser(
        "products",
        help="Manage WooCommerce products",
        description="WooCommerce products subcommands.",
    )
    wc_products_subparsers = wc_products_parser.add_subparsers(
        title="products commands",
        dest="wc_products_command",
        metavar="COMMAND",
    )

    wc_products_list = wc_products_subparsers.add_parser(
        "list",
        help="List WooCommerce products",
        description="Fetch products from a WooCommerce site via the REST API v3.",
    )
    wc_products_list.add_argument(
        "--site",
        type=str,
        required=True,
        help="WooCommerce site ID from woocommerce.sites[] in config",
    )
    wc_products_list.add_argument(
        "--low-stock",
        dest="low_stock",
        action="store_true",
        help="Filter to low-stock and out-of-stock products (client-side filter)",
    )
    wc_products_list.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of products to return (default: 10, max: 100)",
    )
    wc_products_list.set_defaults(
        func=_cli().cmd_wc, wc_command="products", wc_products_command="list"
    )

    wc_products_parser.set_defaults(
        func=_cli().cmd_wc, wc_command="products", wc_products_command="list"
    )

    # wc coupons (write, approval-gated)
    wc_coupons_parser = wc_subparsers.add_parser(
        "coupons",
        help="Manage WooCommerce coupons (write actions, require approval)",
        description="WooCommerce coupon write commands (approval-gated).",
    )
    wc_coupons_subparsers = wc_coupons_parser.add_subparsers(
        title="coupons commands",
        dest="wc_coupons_command",
        metavar="COMMAND",
    )

    wc_coupons_create = wc_coupons_subparsers.add_parser(
        "create",
        help="Create a WooCommerce coupon (requires approval + --yes)",
        description=(
            "Create a new WooCommerce coupon via the REST API v3.\n\n"
            "This is a write action gated by policy approval:\n"
            "  Step 1: Run without --yes to create a pending approval.\n"
            "  Step 2: Approve: rex approvals --approve <id>\n"
            "  Step 3: Re-run with --yes to execute."
        ),
    )
    wc_coupons_create.add_argument(
        "--site",
        type=str,
        required=True,
        help="WooCommerce site ID from woocommerce.sites[] in config",
    )
    wc_coupons_create.add_argument(
        "--code",
        type=str,
        required=True,
        help="Coupon code (non-empty string)",
    )
    wc_coupons_create.add_argument(
        "--amount",
        type=str,
        required=True,
        help="Discount amount (positive number, e.g. '10' or '10.00')",
    )
    wc_coupons_create.add_argument(
        "--type",
        dest="type",
        type=str,
        required=True,
        choices=["percent", "fixed_cart", "fixed_product"],
        help="Discount type: percent, fixed_cart, or fixed_product",
    )
    wc_coupons_create.add_argument(
        "--expires",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Optional expiry date in YYYY-MM-DD format",
    )
    wc_coupons_create.add_argument(
        "--usage-limit",
        dest="usage_limit",
        type=int,
        default=None,
        help="Optional maximum number of times the coupon can be used",
    )
    wc_coupons_create.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Confirm execution (required after approval is granted)",
    )
    wc_coupons_create.add_argument(
        "--user",
        type=str,
        default=None,
        help="Override active user identity for the approval record",
    )
    wc_coupons_create.set_defaults(
        func=_cli().cmd_wc, wc_command="coupons", wc_coupons_command="create"
    )

    wc_coupons_disable = wc_coupons_subparsers.add_parser(
        "disable",
        help="Disable a WooCommerce coupon (requires approval + --yes)",
        description=(
            "Disable a WooCommerce coupon by setting its status to 'draft'.\n\n"
            "This is a write action gated by policy approval:\n"
            "  Step 1: Run without --yes to create a pending approval.\n"
            "  Step 2: Approve: rex approvals --approve <id>\n"
            "  Step 3: Re-run with --yes to execute."
        ),
    )
    wc_coupons_disable.add_argument(
        "--site",
        type=str,
        required=True,
        help="WooCommerce site ID from woocommerce.sites[] in config",
    )
    wc_coupons_disable.add_argument(
        "--coupon-id",
        dest="coupon_id",
        type=int,
        required=True,
        help="WooCommerce coupon ID",
    )
    wc_coupons_disable.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Confirm execution (required after approval is granted)",
    )
    wc_coupons_disable.add_argument(
        "--user",
        type=str,
        default=None,
        help="Override active user identity for the approval record",
    )
    wc_coupons_disable.set_defaults(
        func=_cli().cmd_wc, wc_command="coupons", wc_coupons_command="disable"
    )

    wc_coupons_parser.set_defaults(
        func=_cli().cmd_wc, wc_command="coupons", wc_coupons_command="create"
    )

    wc_parser.set_defaults(func=_cli().cmd_wc, wc_command="orders")
