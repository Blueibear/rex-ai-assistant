"""
US-314: Wire up the Integrations page — verification tests.

Verifies:
- /api/integrations returns all required integrations with name, configured, configure_url
- Home Assistant configure_url is /settings/home-assistant (not Settings > General)
- /api/capabilities endpoint exists and returns a capabilities list
- IntegrationsPage.tsx fetches /api/integrations and /api/capabilities
- No "No integrations found" when integrations are present
- Sidebar route /integrations is registered
- "No capabilities found" section is conditionally rendered (only when empty)
"""

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_gui_app() -> str:
    return (REPO / "rex" / "gui_app.py").read_text(encoding="utf-8")


def read_integrations_page() -> str:
    return (REPO / "gui" / "src" / "pages" / "IntegrationsPage.tsx").read_text(
        encoding="utf-8"
    )


def read_app_layout() -> str:
    return (REPO / "gui" / "src" / "layouts" / "AppLayout.tsx").read_text(
        encoding="utf-8"
    )


def read_app_tsx() -> str:
    return (REPO / "gui" / "src" / "renderer" / "src" / "App.tsx").read_text(
        encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Backend: /api/integrations
# ---------------------------------------------------------------------------


def test_api_integrations_route_exists():
    assert "@app.route(\"/api/integrations\"" in read_gui_app()


def test_api_integrations_returns_list():
    src = read_gui_app()
    assert "integrations = [" in src or '"integrations"' in src


REQUIRED_INTEGRATION_KEYS = [
    "home_assistant",
    "email",
    "calendar",
    "sms",
    "telegram",
    "search",
    "mqtt",
]


@pytest.mark.parametrize("key", REQUIRED_INTEGRATION_KEYS)
def test_required_integration_key_present(key):
    src = read_gui_app()
    assert f'"key": "{key}"' in src or f"'key': '{key}'" in src, (
        f"Integration key '{key}' not found in /api/integrations handler"
    )


def test_home_assistant_configure_url_is_correct():
    """HA configure_url must go to /settings/home-assistant, not /settings."""
    src = read_gui_app()
    ha_block_match = re.search(
        r'"key":\s*"home_assistant".*?"configure_url":\s*"([^"]+)"',
        src,
        re.DOTALL,
    )
    assert ha_block_match is not None, "HA configure_url not found in gui_app.py"
    url = ha_block_match.group(1)
    assert url == "/settings/home-assistant", (
        f"HA configure_url should be /settings/home-assistant, got: {url}"
    )


def test_all_integrations_have_configure_url():
    src = read_gui_app()
    # Every integration dict should include a configure_url key
    assert src.count('"configure_url"') >= len(REQUIRED_INTEGRATION_KEYS)


# ---------------------------------------------------------------------------
# Backend: /api/capabilities
# ---------------------------------------------------------------------------


def test_api_capabilities_route_exists():
    assert "@app.route(\"/api/capabilities\"" in read_gui_app()


def test_api_capabilities_returns_capabilities_key():
    src = read_gui_app()
    assert '"capabilities"' in src


# ---------------------------------------------------------------------------
# Frontend: IntegrationsPage.tsx
# ---------------------------------------------------------------------------


def test_integrations_page_fetches_api_integrations():
    src = read_integrations_page()
    assert "fetch('/api/integrations')" in src or 'fetch("/api/integrations")' in src


def test_integrations_page_fetches_api_capabilities():
    src = read_integrations_page()
    assert "fetch('/api/capabilities')" in src or 'fetch("/api/capabilities")' in src


def test_integrations_page_renders_name_status_configure():
    src = read_integrations_page()
    assert "int.name" in src or "{int.name}" in src
    assert "StatusBadge" in src or "configured" in src
    assert "Configure" in src


def test_integrations_page_no_integrations_message_is_conditional():
    """'No integrations found' should only appear when integrations.length === 0."""
    src = read_integrations_page()
    assert "No integrations found" in src
    # Must be inside a conditional (length check)
    assert "integrations.length === 0" in src or "integrations.length == 0" in src


def test_capabilities_section_is_conditional():
    """Capabilities section must not always render — only when entries exist."""
    src = read_integrations_page()
    # The section should be gated on grouped keys or capabilities length
    assert (
        "Object.keys(grouped).length > 0" in src
        or "capabilities.length" in src
        or "!loading && Object.keys" in src
    )


def test_configure_link_uses_configure_url():
    src = read_integrations_page()
    assert "configure_url" in src
    # Should render a NavLink or anchor pointing to configure_url
    assert "NavLink" in src or "<a" in src


# ---------------------------------------------------------------------------
# Frontend routing
# ---------------------------------------------------------------------------


def test_integrations_route_in_app_layout():
    src = read_app_layout()
    assert "/integrations" in src


def test_integrations_route_in_app_tsx():
    src = read_app_tsx()
    assert "/integrations" in src
    assert "IntegrationsPage" in src


def test_integrations_page_imported_in_app_tsx():
    src = read_app_tsx()
    assert "IntegrationsPage" in src
    assert "IntegrationsPage" in (REPO / "gui" / "src" / "pages" / "IntegrationsPage.tsx").read_text(
        encoding="utf-8"
    )
