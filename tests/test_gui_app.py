from __future__ import annotations

from pathlib import Path

import pytest

from rex import gui_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def app_client():
    app = gui_app._create_flask_app(ui_enabled=True)
    app.config["TESTING"] = True
    return app.test_client()


def test_ui_dist_index_exists_and_is_html() -> None:
    index_path = ROOT / "rex" / "ui" / "dist" / "index.html"

    assert index_path.exists()
    text = index_path.read_text(encoding="utf-8").lower()
    assert "<!doctype html>" in text
    assert "<html" in text
    assert "</html>" in text


def test_dashboard_serves_built_react_index(app_client) -> None:
    response = app_client.get("/ui/")

    assert response.status_code == 200
    assert "text/html" in response.content_type
    assert b"<!doctype html>" in response.data.lower()


def test_main_is_importable() -> None:
    from rex.gui_app import main

    assert callable(main)


def test_gui_port_defaults_and_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REX_GUI_PORT", raising=False)
    assert gui_app._resolve_server_port() == 8765

    monkeypatch.setenv("REX_GUI_PORT", "9000")
    assert gui_app._resolve_server_port() == 9000


def test_gui_port_override_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REX_GUI_PORT", "bad-port")
    with pytest.raises(ValueError, match="REX_GUI_PORT must be an integer"):
        gui_app._resolve_server_port()
