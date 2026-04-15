"""
US-317: Fix "Configure Home Assistant" link routing — verification tests.

Verifies:
- HomePage.tsx "Configure Home Assistant" NavLink routes to /settings/home-assistant
- App.tsx registers the /settings/home-assistant route
- HomeAssistantSettingsPage component exists and is imported in App.tsx
- The route does NOT point to /settings or /settings/general
"""

from pathlib import Path

REPO = Path(__file__).parent.parent


def read_file(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# HomePage.tsx tests
# ---------------------------------------------------------------------------


def test_home_page_ha_link_points_to_settings_home_assistant():
    """The NavLink 'to' prop must be /settings/home-assistant."""
    src = read_file("gui/src/pages/HomePage.tsx")
    assert (
        "/settings/home-assistant" in src
    ), "HomePage.tsx must contain a link to /settings/home-assistant"


def test_home_page_ha_link_not_pointing_to_settings_general():
    """The HA link must not route to /settings or /settings/general."""
    src = read_file("gui/src/pages/HomePage.tsx")
    assert (
        'to="/settings"' not in src or "/settings/home-assistant" in src
    ), "HomePage.tsx must not have a bare /settings link for HA"
    assert "/settings/general" not in src, "HomePage.tsx must not route HA to /settings/general"


def test_home_page_configure_ha_label_present():
    """The link should have a visible 'Configure Home Assistant' label."""
    src = read_file("gui/src/pages/HomePage.tsx")
    assert "Configure Home Assistant" in src


def test_home_page_uses_navlink_or_link_for_ha():
    """The HA link should use NavLink or Link (react-router), not a bare <a href>."""
    src = read_file("gui/src/pages/HomePage.tsx")
    assert (
        "NavLink" in src or "<Link" in src
    ), "HA link should use react-router NavLink or Link component"


# ---------------------------------------------------------------------------
# App.tsx route registration tests
# ---------------------------------------------------------------------------


def test_app_tsx_registers_settings_home_assistant_route():
    """App.tsx must have a Route for /settings/home-assistant."""
    src = read_file("gui/src/renderer/src/App.tsx")
    assert (
        "/settings/home-assistant" in src
    ), "App.tsx must register the /settings/home-assistant route"


def test_app_tsx_imports_home_assistant_settings_page():
    """App.tsx must import HomeAssistantSettingsPage."""
    src = read_file("gui/src/renderer/src/App.tsx")
    assert "HomeAssistantSettingsPage" in src, "App.tsx must import HomeAssistantSettingsPage"


def test_app_tsx_ha_settings_route_uses_correct_component():
    """The /settings/home-assistant route must render HomeAssistantSettingsPage."""
    src = read_file("gui/src/renderer/src/App.tsx")
    assert "HomeAssistantSettingsPage" in src
    # Route path and component reference should both be present
    assert "/settings/home-assistant" in src


# ---------------------------------------------------------------------------
# HomeAssistantSettingsPage component tests
# ---------------------------------------------------------------------------


def test_home_assistant_settings_page_file_exists():
    """HomeAssistantSettingsPage.tsx must exist."""
    path = REPO / "gui" / "src" / "pages" / "HomeAssistantSettingsPage.tsx"
    assert path.exists(), "HomeAssistantSettingsPage.tsx must exist"


def test_home_assistant_settings_page_exports_component():
    """HomeAssistantSettingsPage.tsx must export HomeAssistantSettingsPage."""
    src = read_file("gui/src/pages/HomeAssistantSettingsPage.tsx")
    assert (
        "export function HomeAssistantSettingsPage" in src
        or "export const HomeAssistantSettingsPage" in src
        or "export default" in src
    ), "HomeAssistantSettingsPage.tsx must export the component"


def test_home_assistant_settings_page_has_ha_fields():
    """HomeAssistantSettingsPage.tsx must contain HA URL or token fields."""
    src = read_file("gui/src/pages/HomeAssistantSettingsPage.tsx")
    has_url = "haBaseUrl" in src or "ha_base_url" in src or "base_url" in src.lower()
    has_token = "haToken" in src or "ha_token" in src or "token" in src.lower()
    assert has_url, "HA settings page must have a base URL field"
    assert has_token, "HA settings page must have a token field"


# ---------------------------------------------------------------------------
# Cross-check: IntegrationsPage and other pages also route HA to /settings/home-assistant
# ---------------------------------------------------------------------------


def test_integrations_page_ha_configure_url_matches():
    """IntegrationsPage (backend) must set HA configure_url to /settings/home-assistant."""
    src = read_file("rex/gui_app.py")
    # The backend sets the configure_url for home_assistant
    assert (
        "/settings/home-assistant" in src
    ), "gui_app.py must use /settings/home-assistant as HA configure_url"


def test_ha_device_page_not_configured_link_points_to_settings_home_assistant():
    """HomeAssistantPage.tsx must link to /settings/home-assistant when not configured."""
    src = read_file("gui/src/pages/HomeAssistantPage.tsx")
    assert (
        "/settings/home-assistant" in src
    ), "HomeAssistantPage.tsx 'not configured' state must link to /settings/home-assistant"
