"""Explicit 501 scaffold tests: authenticated, truthful, never fake success.

Matrix rows: FND-018, CAP-008.
"""

from __future__ import annotations

import pytest

from tests.mobile_api.conftest import auth_header, create_user, login_tokens

# Chat, streaming, voice, TTS, and Home Assistant command execution became
# real authenticated routes and are covered by their own test modules.
_SCAFFOLDS = [
    ("get", "/mobile/home/entities"),
    ("get", "/mobile/notifications"),
    ("get", "/mobile/approvals"),
    ("get", "/mobile/tasks"),
    ("get", "/mobile/workflows"),
    ("get", "/mobile/audit-log"),
    ("get", "/mobile/settings"),
]


class TestScaffolds:
    @pytest.mark.parametrize(("method", "path"), _SCAFFOLDS)
    def test_unauthenticated_scaffold_requires_auth(self, client, method: str, path: str) -> None:
        response = getattr(client, method)(path)
        assert response.status_code == 401

    @pytest.mark.parametrize(("method", "path"), _SCAFFOLDS)
    def test_authenticated_scaffold_returns_explicit_501(
        self, client, method: str, path: str
    ) -> None:
        create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        response = getattr(client, method)(path, headers=auth_header(tokens["access_token"]))
        assert response.status_code == 501
        body = response.get_json()
        assert body["error"]["code"] == "NOT_IMPLEMENTED"
        assert body["error"]["retryable"] is False
        assert body["error"]["request_id"]
