"""US-088 external/mobile gateway security contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest


def _build_app(mobile_env: Path, clock, **config_overrides):
    from rex.config import MobileApiConfig
    from rex.mobile_api.app import create_mobile_app
    from rex.mobile_api.db import migrate_users_db
    from rex.mobile_api.services import MobileApiServices

    db_path = mobile_env / "users.db"
    migrate_users_db(db_path)
    services = MobileApiServices.build(
        MobileApiConfig(**config_overrides), db_path=db_path, clock=clock
    )
    app = create_mobile_app(services=services)
    app.config["TESTING"] = True
    return app, services


def test_public_origin_registers_only_mobile_routes(app) -> None:
    routes = {rule.rule for rule in app.url_map.iter_rules() if rule.endpoint != "static"}
    assert routes
    assert all(route.startswith("/mobile/") for route in routes)


def test_local_admin_paths_are_not_present(client) -> None:
    for path in (
        "/api/status/current",
        "/api/admin/permissions/grant",
        "/rex/tools/time_now",
        "/speak",
        "/run",
        "/ui/",
    ):
        response = client.get(path)
        assert response.status_code == 404, path
        assert response.get_json()["error"]["code"] == "NOT_FOUND"


def test_protected_mobile_route_rejects_missing_bearer(client) -> None:
    response = client.get("/mobile/auth/session")
    assert response.status_code == 401
    assert response.get_json()["error"]["code"] == "AUTH_TOKEN_INVALID"


def test_wildcard_cors_is_rejected() -> None:
    from rex.config import MobileApiConfig

    with pytest.raises(ValueError, match="wildcard.*CORS is deny-by-default"):
        MobileApiConfig(allowed_origins=["*"])


def test_cors_allows_only_exact_askrex_origin(mobile_env: Path, clock) -> None:
    app, _ = _build_app(mobile_env, clock, allowed_origins=["https://askrex.app"])
    with app.test_client() as client:
        allowed = client.get("/mobile/status", headers={"Origin": "https://askrex.app"})
        denied = client.get("/mobile/status", headers={"Origin": "https://evil.example"})
    assert allowed.headers.get("Access-Control-Allow-Origin") == "https://askrex.app"
    assert "Access-Control-Allow-Origin" not in denied.headers


def test_public_gateway_rate_limit_is_canonical(mobile_env: Path, clock) -> None:
    app, _ = _build_app(mobile_env, clock, rate_limit_default="2 per minute")
    with app.test_client() as client:
        assert client.get("/mobile/status").status_code == 200
        assert client.get("/mobile/status").status_code == 200
        limited = client.get("/mobile/status")
    assert limited.status_code == 429
    assert limited.get_json()["error"]["code"] == "RATE_LIMITED"
    assert int(limited.headers["Retry-After"]) >= 1


def test_loopback_origin_without_transport_binding_cannot_pair(mobile_env: Path, clock) -> None:
    from rex.mobile_api.pairing import PairingError

    _, services = _build_app(mobile_env, clock)
    assert services.transport_binding is None
    with pytest.raises(PairingError, match="Secure mobile transport is unavailable"):
        services.pairing_authority.create_challenge(user_id="alice", scopes=["chat.send"])


def test_dynamic_openclaw_tool_cannot_gain_mobile_scope() -> None:
    from rex.mobile_api.action_context import required_scope_for_tool

    assert (
        required_scope_for_tool(
            "openclaw_dynamic_tool",
            capability_tags=("chat", "safe", "mobile"),
            operation="read",
        )
        is None
    )


def test_public_gateway_docs_keep_release_gate_explicit() -> None:
    root = Path(__file__).resolve().parents[2]
    files = [
        root / "docs" / "mobile" / "MOBILE_API_THREAT_MODEL.md",
        root / "docs" / "mobile" / "ASKREX_APP_GATEWAY.md",
        root / "docs" / "mobile" / "CLOUDFLARE_TUNNEL.md",
        root / "docs" / "deployment.md",
        root / "docs" / "api.md",
        root / "SECURITY.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in files)
    for required in ("askrex.app", "Cloudflare Tunnel", "CORS", "rate limit", "revocation"):
        assert required.lower() in text.lower()
    assert "public ingress gate is **closed**" in text
