"""Tests for US-126: API reference documentation.

Acceptance criteria:
- docs/api.md or equivalent documents every public endpoint: method, path,
  request schema, response schema, error codes
- authentication requirements documented per endpoint
- at least one example request and response shown per endpoint
- document consistent with the actual running API (no phantom or missing endpoints)
- Typecheck passes
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
API_MD = REPO_ROOT / "docs" / "api.md"

# Endpoints that must appear in the docs. Derived from flask_proxy.py,
# rex/dashboard/routes.py, rex/health.py, and rex_speak_api.py.
REQUIRED_PATHS = [
    "/health/live",
    "/health/ready",
    "/whoami",
    "/search",
    "/contracts",
    "/api/dashboard/status",
    "/api/dashboard/login",
    "/api/dashboard/logout",
    "/api/settings",
    "/api/chat",
    "/api/chat/history",
    "/api/scheduler/jobs",
    "/api/notifications",
    "/api/notifications/stream",
    "/api/voice",
    "/speak",
]

# HTTP methods that must appear
REQUIRED_METHODS = ["GET", "POST", "PATCH", "DELETE"]


# ---------------------------------------------------------------------------
# Document existence
# ---------------------------------------------------------------------------


class TestApiMdExists:
    def test_exists(self) -> None:
        assert API_MD.exists(), "docs/api.md must exist"

    def test_nonempty(self) -> None:
        assert len(API_MD.read_text(encoding="utf-8").strip()) > 500


# ---------------------------------------------------------------------------
# Required endpoints documented
# ---------------------------------------------------------------------------


class TestRequiredEndpoints:
    def _content(self) -> str:
        return API_MD.read_text(encoding="utf-8")

    def test_health_live_documented(self) -> None:
        assert "/health/live" in self._content()

    def test_health_ready_documented(self) -> None:
        assert "/health/ready" in self._content()

    def test_whoami_documented(self) -> None:
        assert "/whoami" in self._content()

    def test_search_documented(self) -> None:
        assert "/search" in self._content()

    def test_dashboard_login_documented(self) -> None:
        assert "/api/dashboard/login" in self._content()

    def test_dashboard_logout_documented(self) -> None:
        assert "/api/dashboard/logout" in self._content()

    def test_settings_documented(self) -> None:
        assert "/api/settings" in self._content()

    def test_chat_documented(self) -> None:
        assert "/api/chat" in self._content()

    def test_chat_history_documented(self) -> None:
        assert "/api/chat/history" in self._content()

    def test_scheduler_jobs_documented(self) -> None:
        assert "/api/scheduler/jobs" in self._content()

    def test_notifications_documented(self) -> None:
        assert "/api/notifications" in self._content()

    def test_notifications_stream_documented(self) -> None:
        assert "/api/notifications/stream" in self._content()

    def test_voice_endpoint_documented(self) -> None:
        assert "/api/voice" in self._content()

    def test_speak_endpoint_documented(self) -> None:
        assert "/speak" in self._content()

    def test_all_required_paths_present(self) -> None:
        content = self._content()
        missing = [p for p in REQUIRED_PATHS if p not in content]
        assert not missing, f"Missing endpoints in docs/api.md: {missing}"


# ---------------------------------------------------------------------------
# HTTP methods documented
# ---------------------------------------------------------------------------


class TestHttpMethods:
    def _content(self) -> str:
        return API_MD.read_text(encoding="utf-8")

    def test_get_method_documented(self) -> None:
        assert "GET" in self._content()

    def test_post_method_documented(self) -> None:
        assert "POST" in self._content()

    def test_patch_method_documented(self) -> None:
        assert "PATCH" in self._content()

    def test_delete_method_documented(self) -> None:
        assert "DELETE" in self._content()


# ---------------------------------------------------------------------------
# Authentication documented per endpoint
# ---------------------------------------------------------------------------


class TestAuthDocumentation:
    def _content(self) -> str:
        return API_MD.read_text(encoding="utf-8")

    def test_bearer_token_mentioned(self) -> None:
        c = self._content().lower()
        assert "bearer" in c or "authorization" in c

    def test_session_auth_mentioned(self) -> None:
        c = self._content().lower()
        assert "session" in c or "cookie" in c

    def test_public_endpoints_identified(self) -> None:
        c = self._content().lower()
        assert "public" in c or "no auth" in c or "none" in c

    def test_authentication_section_present(self) -> None:
        c = self._content().lower()
        assert "authentication" in c or "auth" in c


# ---------------------------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------------------------


class TestSchemas:
    def _content(self) -> str:
        return API_MD.read_text(encoding="utf-8")

    def test_request_body_examples_present(self) -> None:
        """At least one JSON request body example must be shown."""
        c = self._content()
        # Look for JSON code blocks with "message" or "password" field
        assert '"message"' in c or '"password"' in c

    def test_response_examples_present(self) -> None:
        """At least one JSON response example must be shown."""
        c = self._content()
        assert '"status"' in c or '"token"' in c or '"reply"' in c

    def test_error_response_format_documented(self) -> None:
        c = self._content()
        assert '"error"' in c

    def test_error_codes_documented(self) -> None:
        c = self._content()
        # HTTP status codes 400, 401, 404, 500 should be mentioned
        assert "400" in c
        assert "401" in c
        assert "404" in c
        assert "500" in c

    def test_field_table_present(self) -> None:
        """At least one field/parameter table must be present."""
        c = self._content()
        assert "| Field" in c or "| Parameter" in c or "| Name" in c


# ---------------------------------------------------------------------------
# Rate limiting documented
# ---------------------------------------------------------------------------


class TestRateLimitDocumentation:
    def _content(self) -> str:
        return API_MD.read_text(encoding="utf-8")

    def test_rate_limiting_mentioned(self) -> None:
        c = self._content().lower()
        assert "rate limit" in c or "rate-limit" in c or "429" in c

    def test_429_response_shown(self) -> None:
        assert "429" in self._content()

    def test_retry_after_mentioned(self) -> None:
        c = self._content().lower()
        assert "retry-after" in c or "retry after" in c

    def test_health_exemption_mentioned(self) -> None:
        c = self._content().lower()
        assert "exempt" in c or "not rate limit" in c or "excluded" in c


# ---------------------------------------------------------------------------
# Consistency with actual routes
# ---------------------------------------------------------------------------


class TestConsistencyWithCode:
    """Verify documented endpoints match those registered in the source code."""

    def _routes_in_source(self) -> set[str]:
        """Extract route paths from source files."""
        import re

        route_pattern = re.compile(r'@(?:app|dashboard_bp|bp)\.route\("([^"]+)"')
        routes: set[str] = set()

        source_files = [
            REPO_ROOT / "flask_proxy.py",
            REPO_ROOT / "rex" / "dashboard" / "routes.py",
            REPO_ROOT / "rex" / "health.py",
            REPO_ROOT / "rex_speak_api.py",
        ]

        for f in source_files:
            if f.exists():
                for m in route_pattern.finditer(f.read_text(encoding="utf-8")):
                    path = m.group(1)
                    # Normalize parameterised segments
                    path = re.sub(r"<[^>]+>", "{id}", path)
                    routes.add(path)

        return routes

    def test_no_phantom_endpoints_in_docs(self) -> None:
        """Every path in the docs must correspond to a real route (or be a known variant)."""
        content = API_MD.read_text(encoding="utf-8")
        doc_paths = set(re.findall(r"`(/[^`]+)`", content))
        real_routes = self._routes_in_source()

        # Normalise real routes: strip <...> segments
        real_normalized = set()
        for r in real_routes:
            real_normalized.add(r)
            # Also add base path without trailing slash variants
            real_normalized.add(r.rstrip("/"))

        phantom = set()
        for path in doc_paths:
            # Allow parameterised paths like /api/scheduler/jobs/{job_id}
            normalized = re.sub(r"\{[^}]+\}", "{id}", path)
            # Check exact match or partial match (path is a prefix of a real route)
            if not any(
                normalized == r or normalized in r or r in normalized for r in real_normalized
            ):
                phantom.add(path)

        # Filter out paths that are clearly not route paths (e.g. example values)
        phantom = {
            p
            for p in phantom
            if p.startswith("/api")
            or p.startswith("/health")
            or p.startswith("/speak")
            or p.startswith("/search")
            or p.startswith("/whoami")
            or p.startswith("/contracts")
        }

        assert (
            not phantom
        ), f"Paths documented in api.md but not found in source routes: {sorted(phantom)}"
