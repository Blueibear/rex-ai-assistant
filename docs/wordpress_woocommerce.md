# WordPress + WooCommerce Integration

**Implementation Status: Beta (read-only)**

Cycle 6.1 adds read-only monitoring for WordPress sites and WooCommerce stores.
Write actions (order status updates, coupon management) are deferred to Cycle 6.3.

---

## Overview

| Command | Description |
|---|---|
| `rex wp health --site <id>` | Check WordPress site reachability and WP detection |
| `rex wc orders list --site <id>` | List WooCommerce orders (optional status filter) |
| `rex wc products list --site <id>` | List WooCommerce products (optional low-stock filter) |

---

## Configuration

### WordPress

Add a `wordpress.sites[]` section to `config/rex_config.json`:

```json
{
  "wordpress": {
    "sites": [
      {
        "id": "myblog",
        "base_url": "https://example.com",
        "enabled": true,
        "auth_method": "application_password",
        "credential_ref": "wp:myblog",
        "timeout_seconds": 15
      }
    ]
  }
}
```

#### Config keys

| Key | Type | Default | Description |
|---|---|---|---|
| `id` | string | (required) | Unique site identifier used in CLI commands |
| `base_url` | string | (required) | Base URL of the WordPress site (http or https) |
| `enabled` | bool | `true` | Disabled sites are ignored by CLI commands |
| `auth_method` | string | `"none"` | `none`, `application_password`, or `basic` |
| `credential_ref` | string | `""` | CredentialManager key; ignored when `auth_method=none` |
| `timeout_seconds` | int | `15` | HTTP request timeout (1–120 seconds) |

#### CredentialManager refs (WordPress)

When `auth_method` is `application_password` or `basic`, the credential value
must be in `"username:password"` format and stored in `.env` or
`config/credentials.json` — **never** in `rex_config.json`.

Example in `config/credentials.json`:
```json
{
  "wp:myblog": "admin:xxxx xxxx xxxx xxxx xxxx xxxx"
}
```

The application password shown above is a WordPress-generated token created at
**Users → Your Profile → Application Passwords** in the WP admin panel.

### WooCommerce

Add a `woocommerce.sites[]` section to `config/rex_config.json`:

```json
{
  "woocommerce": {
    "sites": [
      {
        "id": "myshop",
        "base_url": "https://example.com",
        "enabled": true,
        "consumer_key_ref": "wc:myshop:key",
        "consumer_secret_ref": "wc:myshop:secret",
        "timeout_seconds": 30
      }
    ]
  }
}
```

#### Config keys

| Key | Type | Default | Description |
|---|---|---|---|
| `id` | string | (required) | Unique site identifier used in CLI commands |
| `base_url` | string | (required) | Base URL of the WordPress/WooCommerce site |
| `enabled` | bool | `true` | Disabled sites are ignored by CLI commands |
| `consumer_key_ref` | string | (required) | CredentialManager key for the WC consumer key |
| `consumer_secret_ref` | string | (required) | CredentialManager key for the WC consumer secret |
| `timeout_seconds` | int | `30` | HTTP request timeout (1–120 seconds) |

#### CredentialManager refs (WooCommerce)

Consumer key and consumer secret are generated in the WooCommerce admin:
**WooCommerce → Settings → Advanced → REST API → Add Key**.

Store them in `.env` or `config/credentials.json`:

```json
{
  "wc:myshop:key": "ck_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "wc:myshop:secret": "cs_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
}
```

Never store raw credentials in `rex_config.json`.

---

## CLI Commands

### WordPress

#### `rex wp health --site <id>`

Check that a WordPress site is reachable and responds like a WordPress instance.

- Calls `GET /wp-json` (public, no auth required).
- If `auth_method` is configured, also calls `GET /wp-json/wp/v2/users/me` to
  verify credentials.
- A 401 on the auth check is reported as `Auth check: FAILED` but does **not**
  fail the overall health result (the site is still reachable).

```
$ rex wp health --site myblog

WordPress Health: myblog
============================================================
  Reachable    : Yes
  WP detected  : Yes
  Site name    : My Blog
  Site URL     : https://example.com
  Auth check   : OK

Overall status: OK
```

### WooCommerce

#### `rex wc orders list --site <id> [--status <status>] [--limit N]`

Fetch orders from a WooCommerce site via REST API v3.

```
$ rex wc orders list --site myshop --status pending --limit 5

WooCommerce Orders: myshop [pending]
============================================================
  #101  [pending]  USD 49.99  2026-02-01  | Alice Smith
  #103  [pending]  USD 12.00  2026-02-03  | Carol Davis

Total: 2 order(s)
```

Available `--status` values: `pending`, `processing`, `on-hold`, `completed`,
`cancelled`, `refunded`, `failed`, `trash` (WooCommerce defaults).

Default `--limit` is 10, maximum is 100.

#### `rex wc products list --site <id> [--low-stock] [--limit N]`

Fetch products from a WooCommerce site via REST API v3.

```
$ rex wc products list --site myshop --low-stock

WooCommerce Products: myshop [low-stock]
============================================================
  #2  Widget B  $19.99  [outofstock]
  #3  Widget C  $4.99  stock=3

Total: 2 product(s)
```

The `--low-stock` filter is applied **client-side** after fetching products.
A product is considered low-stock when:
- `stock_status` is `"outofstock"`, **or**
- `manage_stock=true` and `stock_quantity` ≤ `low_stock_amount`, **or**
- `manage_stock=true` and `stock_quantity` is 0.

Products where `manage_stock=false` or stock fields are missing are excluded
from low-stock results (stock level cannot be determined).

---

## Security notes

- **No secrets in config**: `credential_ref`, `consumer_key_ref`, and
  `consumer_secret_ref` are CredentialManager lookup keys. The actual values
  must be stored in `.env` or `config/credentials.json`.
- **Credentials are never logged**: URL paths, query parameters, and response
  bodies do not appear in log output. Only the site `id` label is logged.
- **Read-only**: This cycle implements only read operations. Write actions
  (updating order status, creating coupons, etc.) are deferred to Cycle 6.3
  and will require explicit policy approval.
- **Timeouts**: All requests have configurable timeouts (default 15 s for WP,
  30 s for WC) to prevent hanging on slow or unreachable sites.

---

## Module layout

```
rex/wordpress/
    __init__.py
    config.py       — Pydantic v2 config models (WordPressSiteConfig, WordPressConfig)
    client.py       — HTTP client (WordPressClient, WPHealthResult)
    service.py      — Service facade (WordPressService, get_wordpress_service)

rex/woocommerce/
    __init__.py
    config.py       — Pydantic v2 config models (WooCommerceSiteConfig, WooCommerceConfig)
    client.py       — HTTP client (WooCommerceClient, OrdersResult, ProductsResult)
    service.py      — Service facade (WooCommerceService, get_woocommerce_service)
```

---

## Deferred to Cycle 6.3

- Write actions: update order status, create/update coupons.
- Policy approval gating for write actions.
- Webhook support for real-time order notifications.
