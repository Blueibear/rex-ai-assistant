"""Status, capabilities, and capability-privacy tests.

Matrix rows: FND-016, FND-017, CAP-001, CAP-009, CAP-010.
"""

from __future__ import annotations

import json


class TestStatus:
    def test_status_is_minimal_and_public(self, client) -> None:
        response = client.get("/mobile/status")
        assert response.status_code == 200
        body = response.get_json()
        assert body["status"] == "ok"
        assert body["api_version"] == "1.0"
        assert body["server_version"]
        assert body["request_id"]
        assert set(body.keys()) == {
            "status",
            "api_version",
            "server_version",
            "request_id",
        }


class TestCapabilities:
    def test_session_one_capabilities_are_truthful(self, client) -> None:
        """CAP-001: authentication true; every runtime feature false."""
        response = client.get("/mobile/capabilities")
        assert response.status_code == 200
        body = response.get_json()
        assert body["api_version"] == "1.0"
        assert body["minimum_app_version"] == "0.1.0"
        features = body["features"]
        assert features["authentication"] is True
        for name in (
            "chat",
            "chat_streaming",
            "websocket_chat",
            "voice_upload",
            "tts",
            "live_voice",
            "notifications",
            "approvals",
            "home_assistant",
        ):
            assert features[name] is False, name

    def test_capabilities_expose_no_sensitive_data(self, client) -> None:
        """CAP-009: no paths, tokens, account IDs, usernames, or model paths."""
        response = client.get("/mobile/capabilities")
        body = response.get_json()
        assert set(body.keys()) == {
            "api_version",
            "minimum_app_version",
            "server_version",
            "features",
        }
        text = json.dumps(body)
        assert "\\\\" not in text  # no Windows paths
        assert "C:/" not in text and "C:\\" not in text
        assert "users.db" not in text
        assert "secret" not in text.lower()
