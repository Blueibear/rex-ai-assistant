"""Unit tests for rex.openclaw.http_client — US-001.

HTTP calls are mocked via unittest.mock so no real network is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rex.openclaw.errors import OpenClawAPIError, OpenClawAuthError, OpenClawConnectionError
from rex.openclaw.http_client import OpenClawClient, get_openclaw_client

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client(max_retries: int = 0) -> OpenClawClient:
    """Return a client pointing at a fake URL with no real retries by default."""
    return OpenClawClient(
        base_url="http://fake-openclaw:18789",
        auth_token="test-token",
        timeout=5,
        max_retries=max_retries,
    )


def _make_response(
    status: int, json_data: dict | None = None, headers: dict | None = None
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    resp.content = b'{"ok": true}' if json_data is None else b"..."
    if json_data is not None:
        resp.json.return_value = json_data
        resp.content = b"..."
    else:
        resp.json.return_value = {"ok": True}
    resp.text = ""
    return resp


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


class TestSuccess:
    def test_post_returns_json(self):
        client = _client()
        ok_resp = _make_response(200, {"answer": 42})
        with patch.object(client._session, "request", return_value=ok_resp) as mock_req:
            result = client.post("/v1/chat/completions", json={"model": "gpt-4"})

        assert result == {"answer": 42}
        mock_req.assert_called_once()
        call_kwargs = mock_req.call_args
        assert call_kwargs.args[0] == "POST"
        assert "/v1/chat/completions" in call_kwargs.args[1]

    def test_get_returns_json(self):
        client = _client()
        ok_resp = _make_response(200, {"status": "ok"})
        with patch.object(client._session, "request", return_value=ok_resp):
            result = client.get("/health")
        assert result == {"status": "ok"}

    def test_patch_returns_json(self):
        client = _client()
        ok_resp = _make_response(200, {"updated": True})
        with patch.object(client._session, "request", return_value=ok_resp):
            result = client.patch("/resource/1", json={"field": "value"})
        assert result == {"updated": True}

    def test_delete_returns_empty_dict_on_204(self):
        client = _client()
        resp = MagicMock()
        resp.status_code = 204
        resp.headers = {}
        resp.content = b""
        resp.text = ""
        with patch.object(client._session, "request", return_value=resp):
            result = client.delete("/resource/1")
        assert result == {}

    def test_authorization_header_sent(self):
        client = OpenClawClient("http://fake:1234", auth_token="secret-abc")
        assert client._session.headers["Authorization"] == "Bearer secret-abc"

    def test_content_type_header_set(self):
        client = OpenClawClient("http://fake:1234", auth_token="tok")
        assert client._session.headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# 401 auth error
# ---------------------------------------------------------------------------


class TestAuthError:
    def test_post_raises_openclaw_auth_error_on_401(self):
        client = _client()
        resp = _make_response(401)
        with patch.object(client._session, "request", return_value=resp):
            with pytest.raises(OpenClawAuthError):
                client.post("/v1/chat/completions", json={})

    def test_auth_error_contains_url(self):
        client = _client()
        resp = _make_response(401)
        with patch.object(client._session, "request", return_value=resp):
            with pytest.raises(OpenClawAuthError) as exc_info:
                client.get("/tools/invoke")
        assert "fake-openclaw" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 429 retry with Retry-After
# ---------------------------------------------------------------------------


class TestRateLimitRetry:
    def test_retries_on_429_and_succeeds(self):
        client = _client(max_retries=3)
        too_many = _make_response(429, headers={"Retry-After": "0"})
        ok_resp = _make_response(200, {"result": "done"})

        with patch.object(client._session, "request", side_effect=[too_many, ok_resp]) as mock_req:
            with patch("time.sleep"):  # don't actually sleep in tests
                result = client.post("/tools/invoke", json={})

        assert result == {"result": "done"}
        assert mock_req.call_count == 2

    def test_raises_api_error_after_max_retries_on_429(self):
        client = _client(max_retries=2)
        too_many = _make_response(429, headers={"Retry-After": "0"})

        with patch.object(client._session, "request", return_value=too_many):
            with patch("time.sleep"):
                with pytest.raises(OpenClawAPIError) as exc_info:
                    client.post("/tools/invoke", json={})

        assert exc_info.value.status == 429

    def test_respects_retry_after_header(self):
        client = _client(max_retries=1)
        too_many = _make_response(429, headers={"Retry-After": "5"})
        ok_resp = _make_response(200, {"ok": True})

        sleep_calls: list[float] = []
        with patch.object(client._session, "request", side_effect=[too_many, ok_resp]):
            with patch("time.sleep", side_effect=lambda t: sleep_calls.append(t)):
                client.post("/path", json={})

        assert sleep_calls == [5.0]


# ---------------------------------------------------------------------------
# 5xx retry
# ---------------------------------------------------------------------------


class TestServerErrorRetry:
    def test_retries_on_500(self):
        client = _client(max_retries=2)
        err_resp = _make_response(500)
        err_resp.text = "internal error"
        ok_resp = _make_response(200, {"ok": True})

        with patch.object(client._session, "request", side_effect=[err_resp, err_resp, ok_resp]):
            with patch("time.sleep"):
                result = client.post("/path", json={})

        assert result == {"ok": True}

    def test_raises_api_error_after_max_retries_on_500(self):
        client = _client(max_retries=1)
        err_resp = _make_response(503)
        err_resp.text = "service unavailable"

        with patch.object(client._session, "request", return_value=err_resp):
            with patch("time.sleep"):
                with pytest.raises(OpenClawAPIError) as exc_info:
                    client.post("/path", json={})

        assert exc_info.value.status == 503


# ---------------------------------------------------------------------------
# Connection error
# ---------------------------------------------------------------------------


class TestConnectionError:
    def test_raises_openclaw_connection_error_on_network_failure(self):
        from requests.exceptions import ConnectionError as RequestsConnectionError

        client = _client()
        with patch.object(
            client._session,
            "request",
            side_effect=RequestsConnectionError("refused"),
        ):
            with pytest.raises(OpenClawConnectionError) as exc_info:
                client.post("/path", json={})

        assert "fake-openclaw" in str(exc_info.value)

    def test_connection_error_wraps_original_cause(self):
        from requests.exceptions import ConnectionError as RequestsConnectionError

        client = _client()
        original = RequestsConnectionError("connection refused")
        with patch.object(client._session, "request", side_effect=original):
            with pytest.raises(OpenClawConnectionError) as exc_info:
                client.get("/health")

        assert exc_info.value.cause is original


# ---------------------------------------------------------------------------
# get_openclaw_client() singleton / None-guard
# ---------------------------------------------------------------------------


class TestGetOpenClawClient:
    def test_returns_none_when_url_is_empty(self):
        config = MagicMock()
        config.openclaw_gateway_url = ""
        assert get_openclaw_client(config) is None

    def test_returns_none_when_url_is_missing(self):
        config = MagicMock(spec=[])  # no attributes at all
        assert get_openclaw_client(config) is None

    def test_returns_client_when_url_is_set(self):
        config = MagicMock()
        config.openclaw_gateway_url = "http://127.0.0.1:18789"
        config.openclaw_gateway_token = "tok"
        config.openclaw_gateway_timeout = 30
        config.openclaw_gateway_max_retries = 3

        from rex.openclaw import http_client

        http_client._CLIENT_CACHE.clear()
        client = get_openclaw_client(config)
        assert client is not None
        assert isinstance(client, OpenClawClient)

    def test_returns_same_instance_for_same_config(self):
        config = MagicMock()
        config.openclaw_gateway_url = "http://127.0.0.1:18789"
        config.openclaw_gateway_token = "tok"
        config.openclaw_gateway_timeout = 30
        config.openclaw_gateway_max_retries = 3

        from rex.openclaw import http_client

        http_client._CLIENT_CACHE.clear()
        c1 = get_openclaw_client(config)
        c2 = get_openclaw_client(config)
        assert c1 is c2
