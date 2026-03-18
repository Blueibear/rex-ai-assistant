"""US-043: Device control — lights and switches via HABridge."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rex.ha_bridge import HABridge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bridge(*, entity_map: dict[str, str] | None = None) -> HABridge:
    """Return a fully mocked HABridge (no real requests or settings needed)."""
    fake_session = MagicMock()
    with patch("rex.ha_bridge.requests"):
        bridge = HABridge(
            base_url="http://homeassistant.local:8123",
            token="test-token",
            secret="",
            entity_map=entity_map or {},
        )
    bridge._session = fake_session
    return bridge


def _ok_response(entity_id: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.content = b'{"result": "ok"}'
    resp.json.return_value = {"result": "ok"}
    return resp


def _error_response() -> Exception:
    import requests as _r

    exc = _r.exceptions.HTTPError("500 Server Error")
    return exc


# ---------------------------------------------------------------------------
# Light control — turn on
# ---------------------------------------------------------------------------


class TestControlLightTurnOn:
    def test_turn_on_returns_success_true(self) -> None:
        bridge = _make_bridge()
        bridge._session.request.return_value = _ok_response("light.kitchen")
        result = bridge.control_light("light.kitchen", "turn_on")
        assert result["success"] is True

    def test_turn_on_message_non_empty(self) -> None:
        bridge = _make_bridge()
        bridge._session.request.return_value = _ok_response("light.kitchen")
        result = bridge.control_light("light.kitchen", "turn_on")
        assert isinstance(result["message"], str)
        assert result["message"]

    def test_turn_on_entity_id_in_result(self) -> None:
        bridge = _make_bridge()
        bridge._session.request.return_value = _ok_response("light.living_room")
        result = bridge.control_light("light.living_room", "turn_on")
        assert result["entity_id"] == "light.living_room"

    def test_turn_on_with_brightness(self) -> None:
        bridge = _make_bridge()
        bridge._session.request.return_value = _ok_response("light.bedroom")
        result = bridge.control_light("light.bedroom", "turn_on", brightness_pct=75)
        assert result["success"] is True
        # Verify brightness_pct was sent in the POST body
        call_kwargs = bridge._session.request.call_args
        assert call_kwargs.kwargs["json"]["brightness_pct"] == 75

    def test_turn_on_brightness_clamped_at_100(self) -> None:
        bridge = _make_bridge()
        bridge._session.request.return_value = _ok_response("light.x")
        bridge.control_light("light.x", "turn_on", brightness_pct=150)
        call_kwargs = bridge._session.request.call_args
        assert call_kwargs.kwargs["json"]["brightness_pct"] == 100

    def test_turn_on_brightness_clamped_at_0(self) -> None:
        bridge = _make_bridge()
        bridge._session.request.return_value = _ok_response("light.x")
        bridge.control_light("light.x", "turn_on", brightness_pct=-10)
        call_kwargs = bridge._session.request.call_args
        assert call_kwargs.kwargs["json"]["brightness_pct"] == 0


# ---------------------------------------------------------------------------
# Light control — turn off
# ---------------------------------------------------------------------------


class TestControlLightTurnOff:
    def test_turn_off_returns_success_true(self) -> None:
        bridge = _make_bridge()
        bridge._session.request.return_value = _ok_response("light.kitchen")
        result = bridge.control_light("light.kitchen", "turn_off")
        assert result["success"] is True

    def test_turn_off_calls_correct_service(self) -> None:
        bridge = _make_bridge()
        bridge._session.request.return_value = _ok_response("light.hall")
        bridge.control_light("light.hall", "turn_off")
        call_kwargs = bridge._session.request.call_args
        assert "/api/services/light/turn_off" in call_kwargs.kwargs["url"]

    def test_turn_on_calls_correct_service(self) -> None:
        bridge = _make_bridge()
        bridge._session.request.return_value = _ok_response("light.hall")
        bridge.control_light("light.hall", "turn_on")
        call_kwargs = bridge._session.request.call_args
        assert "/api/services/light/turn_on" in call_kwargs.kwargs["url"]


# ---------------------------------------------------------------------------
# Light control — failure handling
# ---------------------------------------------------------------------------


class TestControlLightFailures:
    def test_empty_entity_id_raises(self) -> None:
        bridge = _make_bridge()
        with pytest.raises(ValueError, match="entity_id"):
            bridge.control_light("", "turn_on")

    def test_invalid_action_raises(self) -> None:
        bridge = _make_bridge()
        with pytest.raises(ValueError, match="action"):
            bridge.control_light("light.x", "blink")

    def test_ha_error_returns_success_false(self) -> None:
        import requests as _r

        bridge = _make_bridge()
        bridge._session.request.side_effect = _r.exceptions.HTTPError("500")
        result = bridge.control_light("light.kitchen", "turn_on")
        assert result["success"] is False
        assert result["message"]


# ---------------------------------------------------------------------------
# Switch control — turn on
# ---------------------------------------------------------------------------


class TestControlSwitchTurnOn:
    def test_turn_on_returns_success_true(self) -> None:
        bridge = _make_bridge()
        bridge._session.request.return_value = _ok_response("switch.garage")
        result = bridge.control_switch("switch.garage", "turn_on")
        assert result["success"] is True

    def test_turn_on_entity_id_in_result(self) -> None:
        bridge = _make_bridge()
        bridge._session.request.return_value = _ok_response("switch.garage")
        result = bridge.control_switch("switch.garage", "turn_on")
        assert result["entity_id"] == "switch.garage"

    def test_turn_on_calls_correct_service(self) -> None:
        bridge = _make_bridge()
        bridge._session.request.return_value = _ok_response("switch.fan")
        bridge.control_switch("switch.fan", "turn_on")
        call_kwargs = bridge._session.request.call_args
        assert "/api/services/switch/turn_on" in call_kwargs.kwargs["url"]


# ---------------------------------------------------------------------------
# Switch control — turn off
# ---------------------------------------------------------------------------


class TestControlSwitchTurnOff:
    def test_turn_off_returns_success_true(self) -> None:
        bridge = _make_bridge()
        bridge._session.request.return_value = _ok_response("switch.garage")
        result = bridge.control_switch("switch.garage", "turn_off")
        assert result["success"] is True

    def test_turn_off_calls_correct_service(self) -> None:
        bridge = _make_bridge()
        bridge._session.request.return_value = _ok_response("switch.garage")
        bridge.control_switch("switch.garage", "turn_off")
        call_kwargs = bridge._session.request.call_args
        assert "/api/services/switch/turn_off" in call_kwargs.kwargs["url"]

    def test_turn_off_message_non_empty(self) -> None:
        bridge = _make_bridge()
        bridge._session.request.return_value = _ok_response("switch.garage")
        result = bridge.control_switch("switch.garage", "turn_off")
        assert result["message"]


# ---------------------------------------------------------------------------
# Switch control — failure handling
# ---------------------------------------------------------------------------


class TestControlSwitchFailures:
    def test_empty_entity_id_raises(self) -> None:
        bridge = _make_bridge()
        with pytest.raises(ValueError, match="entity_id"):
            bridge.control_switch("", "turn_on")

    def test_invalid_action_raises(self) -> None:
        bridge = _make_bridge()
        with pytest.raises(ValueError, match="action"):
            bridge.control_switch("switch.x", "toggle")

    def test_ha_error_returns_success_false(self) -> None:
        import requests as _r

        bridge = _make_bridge()
        bridge._session.request.side_effect = _r.exceptions.HTTPError("500")
        result = bridge.control_switch("switch.garage", "turn_on")
        assert result["success"] is False
        assert result["message"]
