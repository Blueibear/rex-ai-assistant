"""HTTP client for the WooCommerce REST API v3 (read-only, Cycle 6.1).

API endpoints used
------------------
- ``GET /wp-json/wc/v3/orders``
    List orders.  Supports ``status`` and ``per_page`` query params.
    Response: list of order objects.

- ``GET /wp-json/wc/v3/products``
    List products.  Supports ``per_page`` query param.
    Response: list of product objects.

Authentication
--------------
WooCommerce REST API v3 uses HTTP Basic Auth where:
- Username = consumer key (``ck_...``)
- Password = consumer secret (``cs_...``)

Security notes
--------------
- Consumer key and secret are passed as HTTP Basic Auth and are **never**
  logged.  Only the site ``id`` (label) appears in log output.
- ``requests`` is used (it is a base dependency of Rex).

Dependencies
------------
Uses ``requests`` (already in ``[project.dependencies]``).
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass, field
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Accept header sent with every request.
_ACCEPT_JSON = {"Accept": "application/json"}

# Hard cap on per_page to avoid unreasonably large responses.
MAX_LIMIT = 100

# Stock status value WooCommerce uses for out-of-stock items.
_OUT_OF_STOCK = "outofstock"


@dataclass
class OrdersResult:
    """Result of a ``GET /wc/v3/orders`` call."""

    ok: bool
    orders: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class ProductsResult:
    """Result of a ``GET /wc/v3/products`` call."""

    ok: bool
    products: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class WooCommerceClient:
    """Minimal read-only HTTP client for the WooCommerce REST API v3.

    Parameters
    ----------
    base_url:
        Base URL of the WordPress/WooCommerce site
        (e.g. ``"https://example.com"``).  Trailing slash is stripped.
    consumer_key:
        WooCommerce consumer key (``ck_...``).
    consumer_secret:
        WooCommerce consumer secret (``cs_...``).
    timeout:
        Request timeout in seconds.
    site_id:
        Human-readable label used in log messages (never the credentials).
    """

    def __init__(
        self,
        base_url: str,
        *,
        consumer_key: str,
        consumer_secret: str,
        timeout: int = 30,
        site_id: str = "",
    ) -> None:
        self._base_url = _validate_base_url(base_url)
        # HTTP Basic Auth: consumer key as username, secret as password.
        self._auth = (consumer_key, consumer_secret)
        self._timeout = timeout
        self._label = site_id or self._base_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_orders(
        self,
        *,
        status: str | None = None,
        limit: int = 10,
    ) -> OrdersResult:
        """Fetch a list of orders from WooCommerce.

        Args:
            status: Filter by order status (e.g. ``"pending"``,
                ``"processing"``, ``"completed"``).  ``None`` returns all.
            limit: Maximum number of orders to return (capped at
                :data:`MAX_LIMIT`).

        Returns:
            :class:`OrdersResult` with a list of raw order dicts.
        """
        per_page = max(1, min(limit, MAX_LIMIT))
        params: dict[str, Any] = {"per_page": per_page}
        if status:
            params["status"] = status

        logger.debug(
            "WC list_orders for %s (status=%r, per_page=%d)",
            self._label,
            status,
            per_page,
        )
        try:
            data = self._get("/orders", params=params)
        except Exception as exc:  # noqa: BLE001
            message = _safe_error_message(exc)
            logger.warning("WC list_orders failed for %s: %s", self._label, message)
            return OrdersResult(ok=False, error=message)

        if not isinstance(data, list):
            return OrdersResult(ok=False, error="Unexpected response format from /wc/v3/orders")
        return OrdersResult(ok=True, orders=data)

    def list_products(
        self,
        *,
        limit: int = 10,
        low_stock: bool = False,
    ) -> ProductsResult:
        """Fetch a list of products from WooCommerce.

        Args:
            limit: Maximum number of products to return (capped at
                :data:`MAX_LIMIT`).
            low_stock: When ``True``, filter to products that are low in
                stock (``manage_stock=True`` and ``stock_quantity`` ≤ the
                low-stock-amount, or ``stock_status="outofstock"``).
                Filtering is done client-side on the response because the
                WooCommerce REST API does not natively expose a
                ``low_stock`` filter.

        Returns:
            :class:`ProductsResult` with a list of raw product dicts
            (possibly filtered).
        """
        # When low_stock is requested, fetch a larger batch to have
        # enough candidates after client-side filtering.
        fetch_limit = MAX_LIMIT if low_stock else max(1, min(limit, MAX_LIMIT))
        params: dict[str, Any] = {"per_page": fetch_limit}

        logger.debug(
            "WC list_products for %s (per_page=%d, low_stock=%s)",
            self._label,
            fetch_limit,
            low_stock,
        )
        try:
            data = self._get("/products", params=params)
        except Exception as exc:  # noqa: BLE001
            message = _safe_error_message(exc)
            logger.warning("WC list_products failed for %s: %s", self._label, message)
            return ProductsResult(ok=False, error=message)

        if not isinstance(data, list):
            return ProductsResult(ok=False, error="Unexpected response format from /wc/v3/products")

        products = data
        if low_stock:
            products = _filter_low_stock(products)
            products = products[:limit]

        return ProductsResult(ok=True, products=products)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        """Perform a GET request against the WC API and return parsed JSON.

        Args:
            path: API path relative to ``/wp-json/wc/v3``
                  (e.g. ``"/orders"``).
            params: Optional query parameters.

        Returns:
            Parsed JSON (list or dict).

        Raises:
            requests.HTTPError: On 4xx/5xx responses.
            requests.ConnectionError: If the host is unreachable.
            ValueError: If the response is not valid JSON.
        """
        import requests  # noqa: PLC0415  (base dep, always available)

        url = f"{self._base_url}/wp-json/wc/v3{path}"
        resp = requests.get(
            url,
            headers=_ACCEPT_JSON,
            auth=self._auth,
            params=params,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Helper: client-side low-stock filter
# ---------------------------------------------------------------------------


def _filter_low_stock(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return products that appear to be low in stock or out of stock.

    WooCommerce does not expose a ``low_stock`` filter via the REST API.
    This function applies a best-effort client-side check based on the
    fields returned in the product list response.

    A product is considered low-stock / out-of-stock when **any** of:
    - ``stock_status`` is ``"outofstock"``
    - ``manage_stock`` is ``True`` and ``low_stock_amount`` is set and
      ``stock_quantity`` ≤ ``low_stock_amount``
    - ``manage_stock`` is ``True`` and ``stock_quantity`` is 0

    Products where ``manage_stock`` is ``False`` or the relevant fields are
    absent are excluded (we cannot determine stock level).
    """
    result = []
    for product in products:
        if not isinstance(product, dict):
            continue
        stock_status = product.get("stock_status", "")
        if stock_status == _OUT_OF_STOCK:
            result.append(product)
            continue
        manage_stock = product.get("manage_stock", False)
        if not manage_stock:
            continue
        qty = product.get("stock_quantity")
        if qty is None:
            continue
        try:
            qty = int(qty)
        except (ValueError, TypeError):
            continue
        low_amount = product.get("low_stock_amount")
        if low_amount is not None:
            try:
                low_amount = int(low_amount)
            except (ValueError, TypeError):
                low_amount = None
        if qty == 0 or (low_amount is not None and qty <= low_amount):
            result.append(product)
    return result


def _validate_base_url(base_url: str) -> str:
    """Validate and normalize base_url for safe read-only requests."""
    parsed = urlparse(base_url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        raise ValueError(f"base_url must use http or https scheme, got: {scheme!r}")
    if not parsed.netloc:
        raise ValueError("base_url must include a host (netloc)")
    if parsed.username or parsed.password:
        raise ValueError("base_url must not include embedded credentials")
    _validate_remote_host(parsed.hostname)
    return base_url.rstrip("/")


def _validate_remote_host(hostname: str | None) -> None:
    """Reject localhost/private/reserved targets to reduce SSRF risk."""
    if not hostname:
        raise ValueError("base_url is missing a hostname")

    lowered = hostname.strip().lower()
    if lowered in {"localhost", "localhost.localdomain"}:
        raise ValueError("base_url host must not be localhost or local network")

    try:
        addresses = {ai[4][0] for ai in socket.getaddrinfo(hostname, None)}
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve base_url host: {hostname}") from exc

    for addr in addresses:
        ip = ip_address(addr)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError("base_url host resolves to a local or reserved address")


def _safe_error_message(exc: Exception) -> str:
    """Return a non-sensitive error message for CLI/logging output."""
    import requests  # noqa: PLC0415

    if isinstance(exc, requests.Timeout):
        return "Request timed out"
    if isinstance(exc, requests.HTTPError):
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        return f"HTTP error from WooCommerce API (status={status_code})"
    if isinstance(exc, requests.RequestException):
        return "Request to WooCommerce API failed"
    if isinstance(exc, ValueError):
        return str(exc)
    return "Unexpected error while querying WooCommerce API"


__all__ = [
    "MAX_LIMIT",
    "OrdersResult",
    "ProductsResult",
    "WooCommerceClient",
    "_filter_low_stock",
]
