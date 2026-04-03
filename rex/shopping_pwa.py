"""Shopping list PWA web endpoint (US-SL-004).

Exposes a lightweight, mobile-responsive shopping list at ``/shopping`` that
can be accessed from any phone browser.  Optionally protected by a PIN stored
in ``AppConfig.shopping_pwa_pin``; default is open (no auth).

Endpoints
---------
GET  /shopping               — PWA HTML shell
GET  /shopping/manifest.json — Web app manifest
GET  /shopping/api/items     — JSON: list all items
POST /shopping/api/items     — JSON: add item
PATCH /shopping/api/items/<id> — JSON: toggle checked
POST /shopping/api/clear-checked — JSON: remove all checked items
POST /shopping/pin           — PIN auth form handler
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
from collections.abc import Callable
from functools import wraps
from typing import Any

from flask import Blueprint, Response, make_response, redirect, request

from rex.shopping_list import ShoppingList

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons set by create_blueprint()
# ---------------------------------------------------------------------------
_shopping_list: ShoppingList | None = None
_pin_hash: str | None = None  # SHA-256 hex of the configured PIN, or None
_cookie_secret: str = secrets.token_hex(32)  # ephemeral per-process secret
_COOKIE_NAME = "rex_shopping_auth"
_COOKIE_MAX_AGE = 86400  # 24 h


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _make_token(pin: str) -> str:
    """Return an HMAC-signed token for *pin* using the process-local secret."""
    return hmac.new(_cookie_secret.encode(), pin.encode(), hashlib.sha256).hexdigest()


def _is_authenticated() -> bool:
    """Return True if the request carries a valid auth cookie or no PIN is set."""
    if _pin_hash is None:
        return True
    token = request.cookies.get(_COOKIE_NAME)
    if not token:
        return False
    # Reconstruct expected token from stored PIN hash
    expected = hmac.new(_cookie_secret.encode(), _pin_hash.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(token, expected)


def _login_required(f: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _is_authenticated():
            return _render_pin_page()
        return f(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# HTML / manifest templates (inline — no external templates directory)
# ---------------------------------------------------------------------------

_PIN_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rex Shopping — Sign In</title>
<style>
  body{font-family:system-ui,sans-serif;background:#f5f5f5;display:flex;
       align-items:center;justify-content:center;min-height:100vh;margin:0}
  .card{background:#fff;border-radius:12px;padding:2rem;max-width:320px;
        width:90%;box-shadow:0 2px 8px rgba(0,0,0,.12)}
  h1{margin:0 0 1.5rem;font-size:1.25rem;text-align:center}
  input{width:100%;box-sizing:border-box;padding:.75rem;border:1px solid #ccc;
        border-radius:8px;font-size:1.1rem;margin-bottom:1rem;
        -webkit-text-security:disc}
  button{width:100%;padding:.75rem;background:#2563eb;color:#fff;border:none;
         border-radius:8px;font-size:1rem;cursor:pointer}
  .err{color:#dc2626;font-size:.875rem;margin-top:.5rem;text-align:center}
</style>
</head>
<body>
<div class="card">
  <h1>🛒 Shopping List</h1>
  <form method="post" action="/shopping/pin">
    <input type="tel" name="pin" placeholder="Enter PIN" autofocus
           inputmode="numeric" pattern="[0-9]*">
    <button type="submit">Unlock</button>
    {error}
  </form>
</div>
</body>
</html>"""

_MAIN_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#2563eb">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="Rex Shopping">
<link rel="manifest" href="/shopping/manifest.json">
<title>Rex Shopping List</title>
<style>
  *{box-sizing:border-box}
  body{font-family:system-ui,sans-serif;background:#f5f7fa;margin:0;min-height:100vh;
       padding-bottom:env(safe-area-inset-bottom)}
  header{background:#2563eb;color:#fff;padding:1rem;position:sticky;top:0;z-index:10;
         display:flex;align-items:center;gap:.5rem}
  header h1{margin:0;font-size:1.2rem;flex:1}
  .refresh-btn{background:transparent;border:1px solid rgba(255,255,255,.4);color:#fff;
               border-radius:6px;padding:.25rem .6rem;font-size:.85rem;cursor:pointer}
  main{max-width:600px;margin:0 auto;padding:1rem}
  .add-form{display:flex;gap:.5rem;margin-bottom:1.5rem;flex-wrap:wrap}
  .add-form input[name=name]{flex:1;min-width:140px;padding:.65rem .75rem;border:1px solid #ddd;
                              border-radius:8px;font-size:1rem}
  .add-form input[name=qty]{width:70px;padding:.65rem .5rem;border:1px solid #ddd;
                            border-radius:8px;font-size:1rem;text-align:center}
  .add-form input[name=unit]{width:70px;padding:.65rem .5rem;border:1px solid #ddd;
                             border-radius:8px;font-size:1rem}
  .add-form button{padding:.65rem 1rem;background:#2563eb;color:#fff;border:none;
                   border-radius:8px;font-size:1rem;cursor:pointer;white-space:nowrap}
  section{margin-bottom:1.5rem}
  section h2{font-size:.85rem;text-transform:uppercase;color:#6b7280;letter-spacing:.06em;
             margin:0 0 .5rem}
  ul{list-style:none;margin:0;padding:0}
  li{display:flex;align-items:center;gap:.75rem;background:#fff;border-radius:8px;
     padding:.75rem 1rem;margin-bottom:.5rem;box-shadow:0 1px 3px rgba(0,0,0,.06)}
  li.checked .label{text-decoration:line-through;color:#9ca3af}
  .label{flex:1;font-size:1rem}
  .qty{font-size:.85rem;color:#6b7280;white-space:nowrap}
  input[type=checkbox]{width:20px;height:20px;cursor:pointer;accent-color:#2563eb}
  .clear-btn{width:100%;padding:.65rem;background:#fee2e2;color:#dc2626;border:none;
             border-radius:8px;font-size:.9rem;cursor:pointer;margin-top:.5rem}
  .empty{color:#9ca3af;text-align:center;padding:2rem 0;font-size:.95rem}
  @media(max-width:375px){
    .add-form{gap:.35rem}
    .add-form input[name=qty],.add-form input[name=unit]{width:60px}
  }
</style>
</head>
<body>
<header>
  <h1>🛒 Shopping List</h1>
  <button class="refresh-btn" onclick="load()">Refresh</button>
</header>
<main>
  <form class="add-form" onsubmit="addItem(event)">
    <input name="name" placeholder="Item name" required autocomplete="off">
    <input name="qty" type="number" min="0.01" step="any" value="1" placeholder="Qty">
    <input name="unit" placeholder="unit" autocomplete="off">
    <button type="submit">Add</button>
  </form>
  <section id="tobuy-section">
    <h2>To Buy</h2>
    <ul id="tobuy"></ul>
    <p id="tobuy-empty" class="empty" style="display:none">Nothing here — add something!</p>
  </section>
  <section id="gotit-section" style="display:none">
    <h2>Got It</h2>
    <ul id="gotit"></ul>
    <button class="clear-btn" onclick="clearChecked()">Clear checked</button>
  </section>
</main>
<script>
let _items = [];

async function api(path, opts) {
  const r = await fetch('/shopping/api' + path, {
    headers: {'Content-Type': 'application/json'},
    ...opts
  });
  return r.json();
}

async function load() {
  const data = await api('/items');
  _items = data.items || [];
  render();
}

function render() {
  const unchecked = _items.filter(i => !i.checked);
  const checked   = _items.filter(i =>  i.checked);

  const toBuyEl   = document.getElementById('tobuy');
  const gotItEl   = document.getElementById('gotit');
  const emptyEl   = document.getElementById('tobuy-empty');
  const gotSec    = document.getElementById('gotit-section');

  toBuyEl.innerHTML = unchecked.map(itemHtml).join('');
  gotItEl.innerHTML = checked.map(itemHtml).join('');

  emptyEl.style.display = unchecked.length ? 'none' : '';
  gotSec.style.display   = checked.length  ? ''     : 'none';
}

function itemHtml(item) {
  const qty = item.unit
    ? `${+item.quantity} ${item.unit}`
    : (item.quantity !== 1 ? `${+item.quantity}` : '');
  return `<li class="${item.checked ? 'checked' : ''}">
    <input type="checkbox" ${item.checked ? 'checked' : ''}
           onchange="toggle('${item.id}', this.checked)">
    <span class="label">${esc(item.name)}</span>
    ${qty ? `<span class="qty">${esc(qty)}</span>` : ''}
  </li>`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function toggle(id, checked) {
  await api('/items/' + id, {
    method: 'PATCH',
    body: JSON.stringify({checked})
  });
  await load();
}

async function addItem(e) {
  e.preventDefault();
  const f = e.target;
  await api('/items', {
    method: 'POST',
    body: JSON.stringify({
      name: f.name.value.trim(),
      quantity: parseFloat(f.qty.value) || 1,
      unit: f.unit.value.trim()
    })
  });
  f.name.value = '';
  f.qty.value  = '1';
  f.unit.value = '';
  f.name.focus();
  await load();
}

async function clearChecked() {
  await api('/clear-checked', {method: 'POST'});
  await load();
}

// Initial load + 5-second polling
load();
setInterval(load, 5000);
</script>
</body>
</html>"""

_MANIFEST = {
    "name": "Rex Shopping List",
    "short_name": "Rex Shop",
    "description": "Rex AI shopping list — check items off at the store",
    "start_url": "/shopping",
    "display": "standalone",
    "background_color": "#f5f7fa",
    "theme_color": "#2563eb",
    "icons": [
        {
            "src": "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
            "<text y='.9em' font-size='90'>🛒</text></svg>",
            "sizes": "any",
            "type": "image/svg+xml",
            "purpose": "any maskable",
        }
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_pin_page(error: str = "") -> Response:
    error_html = f'<p class="err">{error}</p>' if error else ""
    html = _PIN_PAGE.replace("{error}", error_html)
    return make_response(html, 200 if not error else 401)


def _json(data: Any, status: int = 200) -> Response:
    return Response(json.dumps(data), status=status, mimetype="application/json")


# ---------------------------------------------------------------------------
# Blueprint factory
# ---------------------------------------------------------------------------


def create_blueprint(shopping_list: ShoppingList, *, pin: str | None = None) -> Blueprint:
    """Return a Flask Blueprint for the shopping list PWA.

    Parameters
    ----------
    shopping_list:
        Shared ``ShoppingList`` instance.
    pin:
        Optional PIN string.  If falsy, the endpoint is unauthenticated.
    """
    global _shopping_list, _pin_hash

    _shopping_list = shopping_list
    _pin_hash = hashlib.sha256(pin.encode()).hexdigest() if pin else None

    bp = Blueprint("shopping_pwa", __name__)

    # ------------------------------------------------------------------
    # PIN auth form handler
    # ------------------------------------------------------------------

    @bp.route("/shopping/pin", methods=["POST"])
    def pin_submit() -> Response:  # type: ignore[return]
        if _pin_hash is None:
            return redirect("/shopping")
        submitted = (request.form.get("pin") or "").strip()
        submitted_hash = hashlib.sha256(submitted.encode()).hexdigest()
        if not hmac.compare_digest(submitted_hash, _pin_hash):
            return _render_pin_page("Incorrect PIN. Please try again.")
        token = hmac.new(_cookie_secret.encode(), _pin_hash.encode(), hashlib.sha256).hexdigest()
        resp = make_response(redirect("/shopping"))
        resp.set_cookie(
            _COOKIE_NAME,
            token,
            max_age=_COOKIE_MAX_AGE,
            httponly=True,
            samesite="Lax",
        )
        return resp

    # ------------------------------------------------------------------
    # PWA shell
    # ------------------------------------------------------------------

    @bp.route("/shopping")
    @_login_required
    def shopping_index() -> Response:  # type: ignore[return]
        return make_response(_MAIN_PAGE, 200, {"Content-Type": "text/html; charset=utf-8"})

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    @bp.route("/shopping/manifest.json")
    def shopping_manifest() -> Response:  # type: ignore[return]
        return _json(_MANIFEST)

    # ------------------------------------------------------------------
    # REST API
    # ------------------------------------------------------------------

    @bp.route("/shopping/api/items", methods=["GET"])
    @_login_required
    def api_list_items() -> Response:  # type: ignore[return]
        assert _shopping_list is not None
        items = [i.to_dict() for i in _shopping_list.list_items()]
        return _json({"items": items})

    @bp.route("/shopping/api/items", methods=["POST"])
    @_login_required
    def api_add_item() -> Response:  # type: ignore[return]
        assert _shopping_list is not None
        body = request.get_json(force=True, silent=True) or {}
        name = (body.get("name") or "").strip()
        if not name:
            return _json({"error": "name is required"}, 400)
        quantity = float(body.get("quantity") or 1)
        unit = (body.get("unit") or "").strip()
        added_by = (body.get("added_by") or "web").strip()
        item = _shopping_list.add_item(name, quantity=quantity, unit=unit, added_by=added_by)
        return _json(item.to_dict(), 201)

    @bp.route("/shopping/api/items/<item_id>", methods=["PATCH"])
    @_login_required
    def api_toggle_item(item_id: str) -> Response:  # type: ignore[return]
        assert _shopping_list is not None
        body = request.get_json(force=True, silent=True) or {}
        checked = bool(body.get("checked", True))
        found = (
            _shopping_list.check_item(item_id) if checked else _shopping_list.uncheck_item(item_id)
        )
        if not found:
            return _json({"error": "item not found"}, 404)
        item = _shopping_list.get_item(item_id)
        return _json(item.to_dict() if item else {"id": item_id})

    @bp.route("/shopping/api/clear-checked", methods=["POST"])
    @_login_required
    def api_clear_checked() -> Response:  # type: ignore[return]
        assert _shopping_list is not None
        removed = _shopping_list.clear_checked()
        return _json({"removed": removed})

    return bp


__all__ = ["create_blueprint"]
