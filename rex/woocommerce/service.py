"""High-level WooCommerce service used by the CLI (Cycle 6.1).

This module bridges config + credentials + the HTTP client into a single,
easy-to-use facade for CLI commands.

Usage example::

    svc = get_woocommerce_service()
    result = svc.list_orders("myshop", status="pending", limit=10)
    print(result)
"""

from __future__ import annotations

import logging
from typing import Any

from rex.woocommerce.client import OrdersResult, ProductsResult, WooCommerceClient, WriteResult
from rex.woocommerce.config import WooCommerceConfig, WooCommerceSiteConfig, load_woocommerce_config

logger = logging.getLogger(__name__)


class WooCommerceSiteNotFoundError(Exception):
    """Raised when a requested site ID is not in the config."""


class WooCommerceSiteDisabledError(Exception):
    """Raised when a requested site is disabled."""


class WooCommerceMissingCredentialError(Exception):
    """Raised when a required credential is not configured."""


class WooCommerceService:
    """Facade over config, credentials, and the WooCommerce HTTP client.

    Parameters
    ----------
    wc_config:
        Parsed :class:`~rex.woocommerce.config.WooCommerceConfig`.
    credential_manager:
        A :class:`~rex.credentials.CredentialManager` instance used to
        resolve credentials from ``consumer_key_ref`` / ``consumer_secret_ref``.
    """

    def __init__(
        self,
        wc_config: WooCommerceConfig,
        credential_manager: Any,
    ) -> None:
        self._config = wc_config
        self._creds = credential_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_orders(
        self,
        site_id: str,
        *,
        status: str | None = None,
        limit: int = 10,
    ) -> OrdersResult:
        """Fetch orders from the specified WooCommerce site.

        Args:
            site_id: The ``id`` field from config.
            status: Optional order status filter (e.g. ``"pending"``).
            limit: Maximum number of orders to return.

        Returns:
            :class:`~rex.woocommerce.client.OrdersResult`.

        Raises:
            :class:`WooCommerceSiteNotFoundError`: If the site ID is unknown.
            :class:`WooCommerceSiteDisabledError`: If the site is disabled.
            :class:`WooCommerceMissingCredentialError`: If a required
                credential is not configured.
        """
        client = self._make_client(site_id)
        return client.list_orders(status=status, limit=limit)

    def list_products(
        self,
        site_id: str,
        *,
        limit: int = 10,
        low_stock: bool = False,
    ) -> ProductsResult:
        """Fetch products from the specified WooCommerce site.

        Args:
            site_id: The ``id`` field from config.
            limit: Maximum number of products to return.
            low_stock: When ``True``, filter to low-stock / out-of-stock
                products (client-side filter).

        Returns:
            :class:`~rex.woocommerce.client.ProductsResult`.

        Raises:
            :class:`WooCommerceSiteNotFoundError`: If the site ID is unknown.
            :class:`WooCommerceSiteDisabledError`: If the site is disabled.
            :class:`WooCommerceMissingCredentialError`: If a required
                credential is not configured.
        """
        client = self._make_client(site_id)
        return client.list_products(limit=limit, low_stock=low_stock)

    # ------------------------------------------------------------------
    # Write API (Cycle 6.3, approval-gated)
    # ------------------------------------------------------------------

    def set_order_status(
        self,
        site_id: str,
        order_id: int,
        *,
        status: str,
    ) -> WriteResult:
        """Update the status of a WooCommerce order.

        Args:
            site_id: The ``id`` field from config.
            order_id: The WooCommerce order ID.
            status: New order status (e.g. ``"completed"``).

        Returns:
            :class:`~rex.woocommerce.client.WriteResult`.

        Raises:
            :class:`WooCommerceSiteNotFoundError`: If the site ID is unknown.
            :class:`WooCommerceSiteDisabledError`: If the site is disabled.
            :class:`WooCommerceMissingCredentialError`: If a required credential
                is not configured.
        """
        client = self._make_client(site_id)
        return client.set_order_status(order_id, status=status)

    def add_order_note(
        self,
        site_id: str,
        order_id: int,
        *,
        note: str,
        customer_note: bool = False,
    ) -> WriteResult:
        """Add a note to a WooCommerce order.

        Args:
            site_id: The ``id`` field from config.
            order_id: The WooCommerce order ID.
            note: The note text.
            customer_note: When ``True``, the note is visible to the customer.

        Returns:
            :class:`~rex.woocommerce.client.WriteResult`.

        Raises:
            :class:`WooCommerceSiteNotFoundError`: If the site ID is unknown.
            :class:`WooCommerceSiteDisabledError`: If the site is disabled.
            :class:`WooCommerceMissingCredentialError`: If a required credential
                is not configured.
        """
        client = self._make_client(site_id)
        return client.add_order_note(order_id, note=note, customer_note=customer_note)

    def create_coupon(
        self,
        site_id: str,
        *,
        code: str,
        amount: str,
        discount_type: str,
        date_expires: str | None = None,
        usage_limit: int | None = None,
    ) -> WriteResult:
        """Create a new WooCommerce coupon.

        Args:
            site_id: The ``id`` field from config.
            code: Coupon code (non-empty string).
            amount: Discount amount as a string (e.g. ``"10"``).
            discount_type: ``"percent"``, ``"fixed_cart"``, or ``"fixed_product"``.
            date_expires: Optional expiry date in ``"YYYY-MM-DD"`` format.
            usage_limit: Optional maximum usage count.

        Returns:
            :class:`~rex.woocommerce.client.WriteResult`.

        Raises:
            :class:`WooCommerceSiteNotFoundError`, :class:`WooCommerceSiteDisabledError`,
            :class:`WooCommerceMissingCredentialError`.
        """
        client = self._make_client(site_id)
        return client.create_coupon(
            code=code,
            amount=amount,
            discount_type=discount_type,
            date_expires=date_expires,
            usage_limit=usage_limit,
        )

    def disable_coupon(self, site_id: str, coupon_id: int) -> WriteResult:
        """Disable a WooCommerce coupon by setting its status to ``"draft"``.

        Args:
            site_id: The ``id`` field from config.
            coupon_id: The WooCommerce coupon ID.

        Returns:
            :class:`~rex.woocommerce.client.WriteResult`.

        Raises:
            :class:`WooCommerceSiteNotFoundError`, :class:`WooCommerceSiteDisabledError`,
            :class:`WooCommerceMissingCredentialError`.
        """
        client = self._make_client(site_id)
        return client.disable_coupon(coupon_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_site(self, site_id: str) -> WooCommerceSiteConfig:
        """Resolve and validate a site config entry.

        Raises:
            :class:`WooCommerceSiteNotFoundError`: unknown ID.
            :class:`WooCommerceSiteDisabledError`: site is disabled.
        """
        cfg = self._config.get_site(site_id)
        if cfg is None:
            raise WooCommerceSiteNotFoundError(
                f"No WooCommerce site with id {site_id!r} found in config. "
                "Add it to the woocommerce.sites[] section of rex_config.json."
            )
        if not cfg.enabled:
            raise WooCommerceSiteDisabledError(
                f"WooCommerce site {site_id!r} is disabled. "
                "Set enabled=true in config/rex_config.json to use it."
            )
        return cfg

    def _resolve_credentials(self, cfg: WooCommerceSiteConfig) -> tuple[str, str]:
        """Resolve consumer key and secret from CredentialManager.

        Returns:
            ``(consumer_key, consumer_secret)`` strings.

        Raises:
            :class:`WooCommerceMissingCredentialError`: If either credential
                is not configured.
        """
        key = self._creds.get_token(cfg.consumer_key_ref)
        if not key:
            raise WooCommerceMissingCredentialError(
                f"Consumer key credential {cfg.consumer_key_ref!r} for "
                f"WooCommerce site {cfg.id!r} is not configured.  "
                "Set it in .env or config/credentials.json."
            )
        secret = self._creds.get_token(cfg.consumer_secret_ref)
        if not secret:
            raise WooCommerceMissingCredentialError(
                f"Consumer secret credential {cfg.consumer_secret_ref!r} for "
                f"WooCommerce site {cfg.id!r} is not configured.  "
                "Set it in .env or config/credentials.json."
            )
        return key, secret

    def _make_client(self, site_id: str) -> WooCommerceClient:
        """Build a :class:`WooCommerceClient` for the given site.

        Raises:
            :class:`WooCommerceSiteNotFoundError`, :class:`WooCommerceSiteDisabledError`,
            :class:`WooCommerceMissingCredentialError`.
        """
        cfg = self._resolve_site(site_id)
        consumer_key, consumer_secret = self._resolve_credentials(cfg)
        return WooCommerceClient(
            base_url=cfg.base_url,
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            timeout=cfg.timeout_seconds,
            site_id=cfg.id,
        )


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_service: WooCommerceService | None = None


def get_woocommerce_service() -> WooCommerceService:
    """Return the module-level :class:`WooCommerceService` singleton.

    Config is loaded from ``rex_config.json`` and credentials from the
    global :func:`~rex.credentials.get_credential_manager`.
    """
    global _service  # noqa: PLW0603
    if _service is None:
        from rex.config_manager import load_config
        from rex.credentials import get_credential_manager

        raw = load_config()
        wc_config = load_woocommerce_config(raw)
        _service = WooCommerceService(
            wc_config=wc_config,
            credential_manager=get_credential_manager(),
        )
    return _service


__all__ = [
    "WooCommerceMissingCredentialError",
    "WooCommerceService",
    "WooCommerceSiteDisabledError",
    "WooCommerceSiteNotFoundError",
    "get_woocommerce_service",
]
