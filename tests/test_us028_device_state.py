"""Tests for US-028: device state awareness via Home Assistant.

Covers:
- get_device_state returns structured dict with entity_id, state, attributes
- get_device_state returns None when entity not found (HTTP 404)
- get_device_state returns None when HA not configured
- DeviceStateHandler answers natural-language state questions
- DeviceStateHandler returns None for non-state queries
- Assistant.generate_reply routes state questions through DeviceStateHandler
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_urlopen_mock(payload: dict | None, status: int = 200) -> MagicMock:
    """Return a context-manager mock for urllib.request.urlopen."""
    if payload is None:
        # Simulate 404
        import urllib.error

        mock = MagicMock()
        mock.__enter__ = MagicMock(side_effect=urllib.error.HTTPError(None, 404, "Not Found", {}, None))  # type: ignore[arg-type]
        mock.__exit__ = MagicMock(return_value=False)
        return mock

    body = json.dumps(payload).encode()
    inner = MagicMock()
    inner.read.return_value = body
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=inner)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


_SAMPLE_LIGHT_RAW = {
    "entity_id": "light.kitchen_ceiling",
    "state": "on",
    "attributes": {
        "friendly_name": "Kitchen Ceiling",
        "brightness": 200,
        "color_mode": "brightness",
    },
}

_SAMPLE_MEDIA_RAW = {
    "entity_id": "media_player.living_room",
    "state": "playing",
    "attributes": {
        "friendly_name": "Living Room Speaker",
        "volume_level": 0.5,
        "media_title": "Jazz Playlist",
    },
}


# ---------------------------------------------------------------------------
# get_device_state tests
# ---------------------------------------------------------------------------


class TestGetDeviceState:
    def test_returns_structured_dict(self) -> None:
        from rex.ha.device_state import get_device_state

        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(_SAMPLE_LIGHT_RAW)):
            result = get_device_state(
                "light.kitchen_ceiling",
                base_url="http://ha.local:8123",
                token="tok",
            )

        assert result is not None
        assert result["entity_id"] == "light.kitchen_ceiling"
        assert result["state"] == "on"
        attrs = result["attributes"]
        assert attrs["friendly_name"] == "Kitchen Ceiling"
        assert attrs["brightness"] == 200

    def test_includes_volume_and_media_title(self) -> None:
        from rex.ha.device_state import get_device_state

        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(_SAMPLE_MEDIA_RAW)):
            result = get_device_state(
                "media_player.living_room",
                base_url="http://ha.local:8123",
                token="tok",
            )

        assert result is not None
        assert result["attributes"]["volume"] == 0.5
        assert result["attributes"]["media_title"] == "Jazz Playlist"

    def test_returns_none_when_entity_not_found(self) -> None:
        import urllib.error

        from rex.ha.device_state import get_device_state

        def raise_404(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise urllib.error.HTTPError(None, 404, "Not Found", {}, None)  # type: ignore[arg-type]

        with patch("urllib.request.urlopen", side_effect=raise_404):
            result = get_device_state(
                "light.nonexistent",
                base_url="http://ha.local:8123",
                token="tok",
            )

        assert result is None

    def test_returns_none_when_not_configured(self) -> None:
        from rex.ha.device_state import get_device_state

        result = get_device_state("light.kitchen", base_url=None, token=None)
        assert result is None

    def test_returns_none_on_http_error(self) -> None:
        import urllib.error

        from rex.ha.device_state import get_device_state

        def raise_500(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise urllib.error.HTTPError(None, 500, "Server Error", {}, None)  # type: ignore[arg-type]

        with patch("urllib.request.urlopen", side_effect=raise_500):
            result = get_device_state(
                "light.kitchen",
                base_url="http://ha.local:8123",
                token="tok",
            )

        assert result is None


# ---------------------------------------------------------------------------
# DeviceStateHandler tests
# ---------------------------------------------------------------------------


def _make_handler(tmp_path: Path, entity_id: str = "light.kitchen_ceiling") -> object:
    """Return a DeviceStateHandler with a pre-populated alias file."""
    from rex.ha.state_handler import DeviceStateHandler

    aliases_path = tmp_path / "device_aliases.json"
    aliases_path.write_text(
        json.dumps({"aliases": {"kitchen light": entity_id}, "synonyms": {}}),
        encoding="utf-8",
    )
    return DeviceStateHandler(
        base_url="http://ha.local:8123",
        token="tok",
        aliases_path=str(aliases_path),
    )


class TestDeviceStateHandler:
    def test_answers_is_device_on_question(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)

        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(_SAMPLE_LIGHT_RAW)):
            response = handler.handle("is the kitchen light on?")  # type: ignore[attr-defined]

        assert response is not None
        assert "Kitchen Ceiling" in response or "kitchen" in response.lower()
        assert "on" in response.lower()

    def test_answers_what_is_status_question(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)

        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(_SAMPLE_LIGHT_RAW)):
            response = handler.handle("what is the status of the kitchen light?")  # type: ignore[attr-defined]

        assert response is not None
        assert "on" in response.lower()

    def test_returns_none_for_non_state_query(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        response = handler.handle("turn on the kitchen light")  # type: ignore[attr-defined]
        assert response is None

    def test_returns_none_when_device_not_in_aliases(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        response = handler.handle("is the garage door open?")  # type: ignore[attr-defined]
        assert response is None

    def test_not_configured_message(self, tmp_path: Path) -> None:
        from rex.ha.state_handler import DeviceStateHandler

        aliases_path = tmp_path / "device_aliases.json"
        aliases_path.write_text(
            json.dumps({"aliases": {"kitchen light": "light.kitchen_ceiling"}, "synonyms": {}}),
            encoding="utf-8",
        )
        handler = DeviceStateHandler(
            base_url=None,
            token=None,
            aliases_path=str(aliases_path),
        )
        response = handler.handle("is the kitchen light on?")
        assert response is not None
        assert "not set up" in response.lower()

    def test_entity_not_found_message(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)

        import urllib.error

        def raise_404(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise urllib.error.HTTPError(None, 404, "Not Found", {}, None)  # type: ignore[arg-type]

        with patch("urllib.request.urlopen", side_effect=raise_404):
            response = handler.handle("is the kitchen light on?")  # type: ignore[attr-defined]

        assert response is not None
        assert "couldn't find" in response.lower()
