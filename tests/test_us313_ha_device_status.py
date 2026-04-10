"""US-313: Home Assistant device status page verification tests."""
from pathlib import Path

HA_PAGE = Path(__file__).parent.parent / "gui" / "src" / "pages" / "HomeAssistantPage.tsx"
HA_SETTINGS_PAGE = (
    Path(__file__).parent.parent / "gui" / "src" / "pages" / "HomeAssistantSettingsPage.tsx"
)
APP_LAYOUT = Path(__file__).parent.parent / "gui" / "src" / "layouts" / "AppLayout.tsx"
APP_TSX = Path(__file__).parent.parent / "gui" / "src" / "renderer" / "src" / "App.tsx"
GUI_APP = Path(__file__).parent.parent / "rex" / "gui_app.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --- Sidebar has Home Assistant nav entry ---

def test_app_layout_has_home_assistant_nav():
    content = read(APP_LAYOUT)
    assert "/home-assistant" in content, "HomeAssistant nav path missing from AppLayout"


def test_app_layout_has_home_assistant_label():
    content = read(APP_LAYOUT)
    assert "Home Assistant" in content, "Home Assistant label missing from sidebar"


# --- Route is registered ---

def test_app_tsx_has_home_assistant_route():
    content = read(APP_TSX)
    assert "/home-assistant" in content, "/home-assistant route missing from App.tsx"


def test_app_tsx_imports_home_assistant_page():
    content = read(APP_TSX)
    assert "HomeAssistantPage" in content, "HomeAssistantPage not imported in App.tsx"


# --- Not-configured state links to /settings/home-assistant (not Settings > General) ---

def test_ha_page_not_configured_links_to_ha_settings():
    content = read(HA_PAGE)
    assert "/settings/home-assistant" in content, (
        "Not-configured state must link to /settings/home-assistant"
    )


def test_ha_page_not_configured_does_not_link_to_general():
    content = read(HA_PAGE)
    # Should not route to generic /settings when HA is unconfigured
    assert "/settings/general" not in content, (
        "Not-configured state must not link to /settings/general"
    )


# --- Device state table columns ---

def test_ha_page_shows_entity_id():
    content = read(HA_PAGE)
    assert "entity_id" in content, "entity_id column missing from device state table"


def test_ha_page_shows_state_column():
    content = read(HA_PAGE)
    assert "state" in content.lower(), "state column missing"


def test_ha_page_shows_last_updated():
    content = read(HA_PAGE)
    assert "last_updated" in content, "last_updated column missing from device state table"


# --- Manual refresh button ---

def test_ha_page_has_refresh_button():
    content = read(HA_PAGE)
    assert "Refresh" in content, "Refresh button missing from HomeAssistantPage"


def test_ha_page_refresh_calls_fetch():
    content = read(HA_PAGE)
    assert "fetchStates" in content, "fetchStates callback missing"


# --- Backend /api/ha/states endpoint ---

def test_backend_ha_states_endpoint_exists():
    content = read(GUI_APP)
    assert "/api/ha/states" in content, "/api/ha/states endpoint missing from gui_app.py"


def test_backend_returns_not_configured_when_no_url():
    content = read(GUI_APP)
    assert "not_configured" in content, "not_configured flag missing from /api/ha/states handler"


def test_backend_returns_entity_id_field():
    content = read(GUI_APP)
    assert '"entity_id"' in content or "'entity_id'" in content, (
        "entity_id field missing from backend states response"
    )


def test_backend_returns_friendly_name():
    content = read(GUI_APP)
    assert "friendly_name" in content, "friendly_name missing from backend states response"
