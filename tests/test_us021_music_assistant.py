"""Tests for US-021: Music Assistant HTTP client."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rex.assistant_errors import IntegrationNotConfiguredError
from rex.integrations.music_assistant import MusicAssistantClient

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _mock_post(return_value: dict | None = None):
    """Return a context manager that patches MusicAssistantClient._post."""
    rv = return_value or {}
    return patch.object(MusicAssistantClient, "_post", return_value=rv)


# ---------------------------------------------------------------------------
# Not-configured guard
# ---------------------------------------------------------------------------


def test_not_configured_raises_on_play():
    client = MusicAssistantClient()
    with pytest.raises(IntegrationNotConfiguredError):
        client.play("jazz")


def test_not_configured_raises_on_pause():
    client = MusicAssistantClient()
    with pytest.raises(IntegrationNotConfiguredError):
        client.pause()


def test_not_configured_raises_on_resume():
    client = MusicAssistantClient()
    with pytest.raises(IntegrationNotConfiguredError):
        client.resume()


def test_not_configured_raises_on_skip():
    client = MusicAssistantClient()
    with pytest.raises(IntegrationNotConfiguredError):
        client.skip()


def test_not_configured_raises_on_set_volume():
    client = MusicAssistantClient()
    with pytest.raises(IntegrationNotConfiguredError):
        client.set_volume(50)


# ---------------------------------------------------------------------------
# Configured — happy-path (mocked HTTP)
# ---------------------------------------------------------------------------


def test_play_posts_correct_payload():
    client = MusicAssistantClient(base_url="http://ma:8095")
    with _mock_post({"ok": True}) as mock_p:
        result = client.play("Shape of You", room="kitchen")
    mock_p.assert_called_once_with(
        "/api/players/play_media", {"query": "Shape of You", "player_id": "kitchen"}
    )
    assert result == {"ok": True}


def test_play_without_room_omits_player_id():
    client = MusicAssistantClient(base_url="http://ma:8095")
    with _mock_post() as mock_p:
        client.play("jazz")
    payload = mock_p.call_args[0][1]
    assert "player_id" not in payload


def test_pause_posts_correct_payload():
    client = MusicAssistantClient(base_url="http://ma:8095")
    with _mock_post() as mock_p:
        client.pause(room="living room")
    mock_p.assert_called_once_with("/api/players/pause", {"player_id": "living room"})


def test_resume_posts_correct_payload():
    client = MusicAssistantClient(base_url="http://ma:8095")
    with _mock_post() as mock_p:
        client.resume(room="bedroom")
    mock_p.assert_called_once_with("/api/players/resume", {"player_id": "bedroom"})


def test_skip_posts_correct_payload():
    client = MusicAssistantClient(base_url="http://ma:8095")
    with _mock_post() as mock_p:
        client.skip()
    mock_p.assert_called_once_with("/api/players/next", {})


def test_set_volume_posts_correct_payload():
    client = MusicAssistantClient(base_url="http://ma:8095")
    with _mock_post() as mock_p:
        client.set_volume(75, room="kitchen")
    mock_p.assert_called_once_with(
        "/api/players/volume_set", {"volume_level": 75, "player_id": "kitchen"}
    )


def test_set_volume_invalid_raises():
    client = MusicAssistantClient(base_url="http://ma:8095")
    with pytest.raises(ValueError, match="0.100"):
        client.set_volume(101)


# ---------------------------------------------------------------------------
# Token header
# ---------------------------------------------------------------------------


def test_token_included_in_headers():
    client = MusicAssistantClient(base_url="http://ma:8095", token="secret")
    headers = client._headers()
    assert headers["Authorization"] == "Bearer secret"


def test_no_token_no_auth_header():
    client = MusicAssistantClient(base_url="http://ma:8095")
    assert "Authorization" not in client._headers()


# ---------------------------------------------------------------------------
# AppConfig fields
# ---------------------------------------------------------------------------


def test_appconfig_has_music_assistant_fields():
    from rex.config import AppConfig

    cfg = AppConfig()
    assert cfg.music_assistant_url is None
    assert cfg.music_assistant_token is None


def test_appconfig_accepts_music_assistant_values():
    from rex.config import AppConfig

    cfg = AppConfig(
        music_assistant_url="http://localhost:8095",
        music_assistant_token="tok",
    )
    assert cfg.music_assistant_url == "http://localhost:8095"
    assert cfg.music_assistant_token == "tok"
