"""Tests for US-048: Plex API client.

Acceptance criteria:
- Plex reachable
- libraries retrieved
- authentication works
- Typecheck passes
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rex.plex_client import (
    PlexAuthError,
    PlexClient,
    PlexConnectionError,
    PlexLibrary,
    PlexMediaItem,
    get_plex_client,
    init_plex_client,
    set_plex_client,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


def _make_client(
    base_url: str = "http://plex.local:32400", token: str = "mytoken"
) -> tuple[PlexClient, MagicMock]:
    """Return a PlexClient with a mock session."""
    session = MagicMock()
    client = PlexClient(base_url=base_url, token=token, session=session)
    return client, session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_global():
    """Isolate global singleton state."""
    set_plex_client(None)
    yield
    set_plex_client(None)


# ---------------------------------------------------------------------------
# Construction and basic attributes
# ---------------------------------------------------------------------------


class TestPlexClientConstruction:
    def test_client_initializes(self):
        client, _ = _make_client()
        assert client is not None

    def test_enabled_when_url_and_token_set(self):
        client, _ = _make_client()
        assert client.enabled is True

    def test_disabled_without_url(self):
        client, _ = _make_client(base_url="", token="token")
        assert client.enabled is False

    def test_disabled_without_token(self):
        client, _ = _make_client(token="")
        assert client.enabled is False

    def test_trailing_slash_stripped_from_url(self):
        client, _ = _make_client(base_url="http://plex.local:32400/")
        assert client._base_url == "http://plex.local:32400"

    def test_token_sent_as_header(self):
        client, session = _make_client(token="secrettoken")
        assert session.headers.update.called
        call_args = session.headers.update.call_args[0][0]
        assert call_args["X-Plex-Token"] == "secrettoken"

    def test_accept_json_header_set(self):
        client, session = _make_client()
        call_args = session.headers.update.call_args[0][0]
        assert call_args["Accept"] == "application/json"


# ---------------------------------------------------------------------------
# Plex reachable (ping / identity)
# ---------------------------------------------------------------------------


class TestPlexReachable:
    def test_ping_returns_true_on_success(self):
        client, session = _make_client()
        session.get.return_value = _make_response({"MediaContainer": {}})
        assert client.ping() is True

    def test_ping_returns_false_on_connection_error(self):
        client, session = _make_client()
        session.get.side_effect = OSError("connection refused")
        assert client.ping() is False

    def test_ping_returns_false_on_auth_error(self):
        client, session = _make_client()
        resp = _make_response({}, status_code=401)
        session.get.return_value = resp
        assert client.ping() is False

    def test_get_raises_connection_error_on_network_failure(self):
        client, session = _make_client()
        session.get.side_effect = OSError("timeout")
        with pytest.raises(PlexConnectionError, match="Cannot reach Plex"):
            client._get("/identity")

    def test_get_raises_auth_error_on_401(self):
        client, session = _make_client()
        resp = _make_response({}, status_code=401)
        session.get.return_value = resp
        with pytest.raises(PlexAuthError, match="token"):
            client._get("/identity")


# ---------------------------------------------------------------------------
# Libraries retrieved
# ---------------------------------------------------------------------------


class TestPlexLibraries:
    def _library_response(self) -> dict:
        return {
            "MediaContainer": {
                "Directory": [
                    {
                        "key": "1",
                        "title": "Movies",
                        "type": "movie",
                        "count": 42,
                    },
                    {
                        "key": "2",
                        "title": "TV Shows",
                        "type": "show",
                        "count": 10,
                    },
                ]
            }
        }

    def test_get_libraries_returns_list(self):
        client, session = _make_client()
        session.get.return_value = _make_response(self._library_response())
        libs = client.get_libraries()
        assert isinstance(libs, list)
        assert len(libs) == 2

    def test_library_fields_populated(self):
        client, session = _make_client()
        session.get.return_value = _make_response(self._library_response())
        libs = client.get_libraries()
        movies = libs[0]
        assert isinstance(movies, PlexLibrary)
        assert movies.library_id == "1"
        assert movies.title == "Movies"
        assert movies.library_type == "movie"
        assert movies.count == 42

    def test_get_libraries_empty_when_no_sections(self):
        client, session = _make_client()
        session.get.return_value = _make_response({"MediaContainer": {}})
        libs = client.get_libraries()
        assert libs == []

    def test_get_libraries_uses_correct_endpoint(self):
        client, session = _make_client()
        session.get.return_value = _make_response({"MediaContainer": {"Directory": []}})
        client.get_libraries()
        url = session.get.call_args[0][0]
        assert "/library/sections" in url


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestPlexAuthentication:
    def test_authenticated_request_succeeds(self):
        client, session = _make_client(token="validtoken")
        session.get.return_value = _make_response({"MediaContainer": {"Directory": []}})
        libs = client.get_libraries()
        assert isinstance(libs, list)

    def test_invalid_token_raises_auth_error(self):
        client, session = _make_client(token="badtoken")
        resp = _make_response({}, status_code=401)
        session.get.return_value = resp
        with pytest.raises(PlexAuthError):
            client.get_libraries()

    def test_connection_error_handled_gracefully(self):
        client, session = _make_client()
        session.get.side_effect = OSError("connection refused")
        with pytest.raises(PlexConnectionError):
            client.get_libraries()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestPlexSearch:
    def _search_response(self) -> dict:
        return {
            "MediaContainer": {
                "Metadata": [
                    {
                        "ratingKey": "100",
                        "title": "The Matrix",
                        "type": "movie",
                        "year": 1999,
                        "summary": "A sci-fi film.",
                        "duration": 8160000,
                    }
                ]
            }
        }

    def test_search_returns_items(self):
        client, session = _make_client()
        session.get.return_value = _make_response(self._search_response())
        results = client.search("matrix")
        assert len(results) == 1
        assert isinstance(results[0], PlexMediaItem)

    def test_search_item_fields(self):
        client, session = _make_client()
        session.get.return_value = _make_response(self._search_response())
        item = client.search("matrix")[0]
        assert item.rating_key == "100"
        assert item.title == "The Matrix"
        assert item.year == 1999
        assert item.media_type == "movie"

    def test_search_empty_query_returns_empty_list(self):
        client, _ = _make_client()
        results = client.search("")
        assert results == []

    def test_search_no_results_returns_empty_list(self):
        client, session = _make_client()
        session.get.return_value = _make_response({"MediaContainer": {}})
        results = client.search("nothing")
        assert results == []


# ---------------------------------------------------------------------------
# Global singleton helpers
# ---------------------------------------------------------------------------


class TestGlobalHelpers:
    def test_get_plex_client_returns_none_by_default(self):
        assert get_plex_client() is None

    def test_set_plex_client_stores_client(self):
        client, _ = _make_client()
        set_plex_client(client)
        assert get_plex_client() is client

    def test_init_plex_client_creates_and_stores(self):
        session = MagicMock()
        import rex.plex_client as mod

        original = mod._requests_module.Session
        mod._requests_module.Session = lambda: session  # type: ignore[union-attr]
        try:
            client = init_plex_client("http://plex.local:32400", "tok123")
            assert client is not None
            assert get_plex_client() is client
        finally:
            mod._requests_module.Session = original  # type: ignore[union-attr]

    def test_set_plex_client_none_clears_global(self):
        client, _ = _make_client()
        set_plex_client(client)
        set_plex_client(None)
        assert get_plex_client() is None
