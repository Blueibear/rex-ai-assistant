"""US-049: Plex playback control tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from rex.plex_client import PlexClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> tuple[PlexClient, MagicMock]:
    """Return a PlexClient with a mocked requests session."""
    mock_session = MagicMock()
    client = PlexClient(
        base_url="http://plex.local:32400",
        token="test-token",
        session=mock_session,
    )
    return client, mock_session


def _ok_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    return resp


def _error_response(status: int) -> MagicMock:
    from requests.exceptions import HTTPError

    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status.side_effect = HTTPError(response=resp)
    return resp


# ---------------------------------------------------------------------------
# play() tests
# ---------------------------------------------------------------------------


class TestPlay:
    def test_play_returns_true_on_success(self) -> None:
        client, session = _make_client()
        session.get.return_value = _ok_response()
        assert client.play("my-client-id") is True

    def test_play_calls_correct_endpoint(self) -> None:
        client, session = _make_client()
        session.get.return_value = _ok_response()
        client.play("my-client-id")
        url = session.get.call_args[0][0]
        assert url == "http://plex.local:32400/player/playback/play"

    def test_play_sends_client_id_in_params(self) -> None:
        client, session = _make_client()
        session.get.return_value = _ok_response()
        client.play("PLAYER-XYZ")
        params = session.get.call_args[1]["params"]
        assert params["X-Plex-Target-Client-Identifier"] == "PLAYER-XYZ"

    def test_play_with_rating_key_includes_key_params(self) -> None:
        client, session = _make_client()
        session.get.return_value = _ok_response()
        client.play("player-1", rating_key="12345")
        params = session.get.call_args[1]["params"]
        assert params["key"] == "/library/metadata/12345"
        assert params["containerKey"] == "/library/metadata/12345"
        assert params["type"] == "video"

    def test_play_without_rating_key_omits_key_params(self) -> None:
        client, session = _make_client()
        session.get.return_value = _ok_response()
        client.play("player-1")
        params = session.get.call_args[1]["params"]
        assert "key" not in params
        assert "containerKey" not in params

    def test_play_returns_false_on_connection_error(self) -> None:
        client, session = _make_client()
        session.get.side_effect = OSError("connection refused")
        assert client.play("player-1") is False

    def test_play_returns_false_on_auth_error(self) -> None:
        client, session = _make_client()
        session.get.return_value = _error_response(401)
        assert client.play("player-1") is False

    def test_play_returns_false_on_server_error(self) -> None:
        client, session = _make_client()
        session.get.return_value = _error_response(500)
        assert client.play("player-1") is False

    def test_play_custom_command_id(self) -> None:
        client, session = _make_client()
        session.get.return_value = _ok_response()
        client.play("player-1", command_id=42)
        params = session.get.call_args[1]["params"]
        assert params["commandID"] == 42


# ---------------------------------------------------------------------------
# pause() tests
# ---------------------------------------------------------------------------


class TestPause:
    def test_pause_returns_true_on_success(self) -> None:
        client, session = _make_client()
        session.get.return_value = _ok_response()
        assert client.pause("my-client-id") is True

    def test_pause_calls_correct_endpoint(self) -> None:
        client, session = _make_client()
        session.get.return_value = _ok_response()
        client.pause("my-client-id")
        url = session.get.call_args[0][0]
        assert url == "http://plex.local:32400/player/playback/pause"

    def test_pause_sends_client_id_in_params(self) -> None:
        client, session = _make_client()
        session.get.return_value = _ok_response()
        client.pause("PLAYER-ABC")
        params = session.get.call_args[1]["params"]
        assert params["X-Plex-Target-Client-Identifier"] == "PLAYER-ABC"

    def test_pause_returns_false_on_connection_error(self) -> None:
        client, session = _make_client()
        session.get.side_effect = OSError("timeout")
        assert client.pause("player-1") is False

    def test_pause_returns_false_on_auth_error(self) -> None:
        client, session = _make_client()
        session.get.return_value = _error_response(401)
        assert client.pause("player-1") is False

    def test_pause_returns_false_on_server_error(self) -> None:
        client, session = _make_client()
        session.get.return_value = _error_response(503)
        assert client.pause("player-1") is False

    def test_pause_custom_command_id(self) -> None:
        client, session = _make_client()
        session.get.return_value = _ok_response()
        client.pause("player-1", command_id=7)
        params = session.get.call_args[1]["params"]
        assert params["commandID"] == 7


# ---------------------------------------------------------------------------
# stop() tests
# ---------------------------------------------------------------------------


class TestStop:
    def test_stop_returns_true_on_success(self) -> None:
        client, session = _make_client()
        session.get.return_value = _ok_response()
        assert client.stop("my-client-id") is True

    def test_stop_calls_correct_endpoint(self) -> None:
        client, session = _make_client()
        session.get.return_value = _ok_response()
        client.stop("my-client-id")
        url = session.get.call_args[0][0]
        assert url == "http://plex.local:32400/player/playback/stop"

    def test_stop_sends_client_id_in_params(self) -> None:
        client, session = _make_client()
        session.get.return_value = _ok_response()
        client.stop("PLAYER-DEF")
        params = session.get.call_args[1]["params"]
        assert params["X-Plex-Target-Client-Identifier"] == "PLAYER-DEF"

    def test_stop_returns_false_on_connection_error(self) -> None:
        client, session = _make_client()
        session.get.side_effect = OSError("network unreachable")
        assert client.stop("player-1") is False

    def test_stop_returns_false_on_auth_error(self) -> None:
        client, session = _make_client()
        session.get.return_value = _error_response(401)
        assert client.stop("player-1") is False

    def test_stop_returns_false_on_server_error(self) -> None:
        client, session = _make_client()
        session.get.return_value = _error_response(404)
        assert client.stop("player-1") is False

    def test_stop_custom_command_id(self) -> None:
        client, session = _make_client()
        session.get.return_value = _ok_response()
        client.stop("player-1", command_id=99)
        params = session.get.call_args[1]["params"]
        assert params["commandID"] == 99
