"""Offline tests for the Rex Windows Agent server (Cycle 5.3).

All tests use the Flask test client — no real subprocess execution, no network,
no admin rights, and no Windows-specific behaviour required.

Coverage targets
----------------
- 401 returned when ``X-Auth-Token`` header is missing
- 401 returned when ``X-Auth-Token`` value is wrong
- 200 returned for all three endpoints when token is correct
- 403 returned when command is not on the allowlist (and no subprocess called)
- 200 with correct JSON when command is allowlisted (subprocess stubbed)
- 429 returned after the rate limit is exceeded
- Output truncation: stdout/stderr are capped at ``max_output`` bytes
- JSON response schema validation for ``/run``
- ``/health`` response schema
- ``/status`` response schema
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("flask")

from rex.computers.agent_server import AUTH_HEADER, create_app  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TEST_TOKEN = "test-secret-token-for-ci"
_ALLOWED_CMDS = frozenset({"whoami", "echo"})


@pytest.fixture()
def app():
    """Flask app with a fixed token, small allowlist, and no rate limiting."""
    return create_app(
        token=_TEST_TOKEN,
        allowlist=_ALLOWED_CMDS,
        rate_limit=0,  # disable rate limiting for most tests
        cmd_timeout=5,
        max_output=1024,
    )


@pytest.fixture()
def client(app):
    """Flask test client."""
    return app.test_client()


def _auth_headers(token: str = _TEST_TOKEN) -> dict[str, str]:
    return {AUTH_HEADER: token}


# ===========================================================================
# Auth tests
# ===========================================================================


class TestAuth:
    def test_health_missing_token_returns_401(self, client) -> None:
        resp = client.get("/health")
        assert resp.status_code == 401
        data = resp.get_json()
        assert "error" in data

    def test_health_wrong_token_returns_401(self, client) -> None:
        resp = client.get("/health", headers={AUTH_HEADER: "wrong-token"})
        assert resp.status_code == 401

    def test_status_missing_token_returns_401(self, client) -> None:
        resp = client.get("/status")
        assert resp.status_code == 401

    def test_run_missing_token_returns_401(self, client) -> None:
        resp = client.post(
            "/run",
            data=json.dumps({"command": "whoami", "args": []}),
            content_type="application/json",
        )
        assert resp.status_code == 401

    def test_run_wrong_token_returns_401(self, client) -> None:
        resp = client.post(
            "/run",
            data=json.dumps({"command": "whoami", "args": []}),
            content_type="application/json",
            headers={AUTH_HEADER: "not-the-right-token"},
        )
        assert resp.status_code == 401

    def test_options_health_requires_token(self, client) -> None:
        resp = client.open("/health", method="OPTIONS")
        assert resp.status_code == 401

    def test_options_run_requires_token(self, client) -> None:
        resp = client.open("/run", method="OPTIONS")
        assert resp.status_code == 401

    def test_options_run_wrong_token_returns_401(self, client) -> None:
        resp = client.open("/run", method="OPTIONS", headers={AUTH_HEADER: "bad-token"})
        assert resp.status_code == 401

    def test_options_run_valid_token_is_allowed(self, client) -> None:
        resp = client.open("/run", method="OPTIONS", headers=_auth_headers())
        assert resp.status_code == 200


# ===========================================================================
# /health tests
# ===========================================================================


class TestHealth:
    def test_health_ok(self, client) -> None:
        resp = client.get("/health", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {"status": "ok"}

    def test_health_content_type_json(self, client) -> None:
        resp = client.get("/health", headers=_auth_headers())
        assert "application/json" in resp.content_type


# ===========================================================================
# /status tests
# ===========================================================================


class TestStatus:
    def test_status_ok(self, client) -> None:
        resp = client.get("/status", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.get_json()
        assert "hostname" in data
        assert "os" in data
        assert "user" in data
        assert "time" in data

    def test_status_time_format(self, client) -> None:
        resp = client.get("/status", headers=_auth_headers())
        data = resp.get_json()
        # Expect ISO-8601 datetime string
        t = data["time"]
        assert "T" in t, f"Expected ISO datetime, got: {t!r}"

    def test_status_fields_are_strings(self, client) -> None:
        resp = client.get("/status", headers=_auth_headers())
        data = resp.get_json()
        for field in ("hostname", "os", "user", "time"):
            assert isinstance(data[field], str), f"{field!r} should be a string"


# ===========================================================================
# /run allowlist tests
# ===========================================================================


class TestAllowlist:
    def test_run_disallowed_command_returns_403(self, client) -> None:
        resp = client.post(
            "/run",
            data=json.dumps({"command": "del", "args": []}),
            content_type="application/json",
            headers=_auth_headers(),
        )
        assert resp.status_code == 403
        data = resp.get_json()
        assert "error" in data
        assert "del" in data["error"] or "allowlist" in data["error"].lower()

    def test_run_disallowed_command_does_not_call_subprocess(self, client) -> None:
        with patch("rex.computers.agent_server.subprocess") as mock_sub:
            resp = client.post(
                "/run",
                data=json.dumps({"command": "rm", "args": ["-rf", "/"]}),
                content_type="application/json",
                headers=_auth_headers(),
            )
        assert resp.status_code == 403
        mock_sub.run.assert_not_called()

    def test_run_empty_command_returns_400(self, client) -> None:
        resp = client.post(
            "/run",
            data=json.dumps({"command": "", "args": []}),
            content_type="application/json",
            headers=_auth_headers(),
        )
        assert resp.status_code == 400

    def test_run_missing_command_field_returns_400(self, client) -> None:
        resp = client.post(
            "/run",
            data=json.dumps({"args": []}),
            content_type="application/json",
            headers=_auth_headers(),
        )
        assert resp.status_code == 400

    def test_run_non_json_body_returns_400(self, client) -> None:
        resp = client.post(
            "/run",
            data=b"not json",
            content_type="application/json",
            headers=_auth_headers(),
        )
        assert resp.status_code == 400


# ===========================================================================
# /run execution tests (subprocess stubbed)
# ===========================================================================

_FAKE_COMPLETED = MagicMock()
_FAKE_COMPLETED.returncode = 0
_FAKE_COMPLETED.stdout = b"alice\n"
_FAKE_COMPLETED.stderr = b""


class TestRunExecution:
    def test_run_allowlisted_command_returns_200(self, client) -> None:
        with patch("rex.computers.agent_server.subprocess.run", return_value=_FAKE_COMPLETED):
            resp = client.post(
                "/run",
                data=json.dumps({"command": "whoami", "args": []}),
                content_type="application/json",
                headers=_auth_headers(),
            )
        assert resp.status_code == 200

    def test_run_response_schema(self, client) -> None:
        with patch("rex.computers.agent_server.subprocess.run", return_value=_FAKE_COMPLETED):
            resp = client.post(
                "/run",
                data=json.dumps({"command": "whoami", "args": []}),
                content_type="application/json",
                headers=_auth_headers(),
            )
        data = resp.get_json()
        assert "exit_code" in data
        assert "stdout" in data
        assert "stderr" in data
        assert "duration_ms" in data

    def test_run_returns_stdout(self, client) -> None:
        with patch("rex.computers.agent_server.subprocess.run", return_value=_FAKE_COMPLETED):
            resp = client.post(
                "/run",
                data=json.dumps({"command": "whoami", "args": []}),
                content_type="application/json",
                headers=_auth_headers(),
            )
        data = resp.get_json()
        assert data["exit_code"] == 0
        assert "alice" in data["stdout"]

    def test_run_invokes_subprocess_with_shell_false(self, client) -> None:
        with patch(
            "rex.computers.agent_server.subprocess.run", return_value=_FAKE_COMPLETED
        ) as mock_run:
            client.post(
                "/run",
                data=json.dumps({"command": "whoami", "args": ["--all"]}),
                content_type="application/json",
                headers=_auth_headers(),
            )
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs.get("shell") is False, "subprocess must not use shell=True"

    def test_run_argv_construction(self, client) -> None:
        """Command and args are assembled into a single argv list."""
        with patch(
            "rex.computers.agent_server.subprocess.run", return_value=_FAKE_COMPLETED
        ) as mock_run:
            client.post(
                "/run",
                data=json.dumps({"command": "echo", "args": ["hello", "world"]}),
                content_type="application/json",
                headers=_auth_headers(),
            )
        argv = mock_run.call_args.args[0]
        assert argv == ["echo", "hello", "world"]

    def test_run_nonzero_exit_code(self, client) -> None:
        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stdout = b""
        fake_result.stderr = b"error output"

        with patch("rex.computers.agent_server.subprocess.run", return_value=fake_result):
            resp = client.post(
                "/run",
                data=json.dumps({"command": "whoami", "args": []}),
                content_type="application/json",
                headers=_auth_headers(),
            )
        data = resp.get_json()
        assert data["exit_code"] == 1
        assert "error output" in data["stderr"]

    def test_run_duration_ms_is_non_negative_int(self, client) -> None:
        with patch("rex.computers.agent_server.subprocess.run", return_value=_FAKE_COMPLETED):
            resp = client.post(
                "/run",
                data=json.dumps({"command": "whoami", "args": []}),
                content_type="application/json",
                headers=_auth_headers(),
            )
        data = resp.get_json()
        assert isinstance(data["duration_ms"], int)
        assert data["duration_ms"] >= 0


# ===========================================================================
# Output truncation tests
# ===========================================================================


class TestOutputTruncation:
    def test_stdout_truncated_to_max_output(self) -> None:
        app = create_app(
            token=_TEST_TOKEN,
            allowlist=_ALLOWED_CMDS,
            rate_limit=0,
            max_output=10,
        )
        client = app.test_client()

        large_output = MagicMock()
        large_output.returncode = 0
        large_output.stdout = b"A" * 500  # 500 bytes, max is 10
        large_output.stderr = b""

        with patch("rex.computers.agent_server.subprocess.run", return_value=large_output):
            resp = client.post(
                "/run",
                data=json.dumps({"command": "whoami", "args": []}),
                content_type="application/json",
                headers=_auth_headers(),
            )
        data = resp.get_json()
        assert len(data["stdout"]) == 10

    def test_stderr_truncated_to_max_output(self) -> None:
        app = create_app(
            token=_TEST_TOKEN,
            allowlist=_ALLOWED_CMDS,
            rate_limit=0,
            max_output=8,
        )
        client = app.test_client()

        large_stderr = MagicMock()
        large_stderr.returncode = 1
        large_stderr.stdout = b""
        large_stderr.stderr = b"E" * 200  # 200 bytes, max is 8

        with patch("rex.computers.agent_server.subprocess.run", return_value=large_stderr):
            resp = client.post(
                "/run",
                data=json.dumps({"command": "whoami", "args": []}),
                content_type="application/json",
                headers=_auth_headers(),
            )
        data = resp.get_json()
        assert len(data["stderr"]) == 8


# ===========================================================================
# Rate limiting tests
# ===========================================================================


class TestRateLimiting:
    def test_rate_limit_returns_429(self) -> None:
        app = create_app(
            token=_TEST_TOKEN,
            allowlist=_ALLOWED_CMDS,
            rate_limit=3,  # allow only 3 requests per minute
        )
        client = app.test_client()
        headers = _auth_headers()

        # First 3 requests should succeed.
        for _ in range(3):
            resp = client.get("/health", headers=headers)
            assert resp.status_code == 200, "Expected 200 within rate limit"

        # The 4th request should be rate-limited.
        resp = client.get("/health", headers=headers)
        assert resp.status_code == 429
        data = resp.get_json()
        assert "error" in data

    def test_rate_limit_error_body_is_json(self) -> None:
        app = create_app(
            token=_TEST_TOKEN,
            allowlist=_ALLOWED_CMDS,
            rate_limit=1,
        )
        client = app.test_client()
        headers = _auth_headers()

        client.get("/health", headers=headers)  # consume the 1 allowed
        resp = client.get("/health", headers=headers)
        assert resp.status_code == 429
        data = resp.get_json()
        assert isinstance(data, dict)

    def test_rate_limit_zero_disables_limiting(self) -> None:
        app = create_app(
            token=_TEST_TOKEN,
            allowlist=_ALLOWED_CMDS,
            rate_limit=0,
        )
        client = app.test_client()
        headers = _auth_headers()

        for _ in range(20):
            resp = client.get("/health", headers=headers)
            assert resp.status_code == 200


# ===========================================================================
# Subprocess timeout / not-found edge cases
# ===========================================================================


class TestSubprocessEdgeCases:
    def test_timeout_returns_200_with_error_in_stderr(self, client) -> None:
        import subprocess as _sp

        with patch(
            "rex.computers.agent_server.subprocess.run",
            side_effect=_sp.TimeoutExpired(cmd=["whoami"], timeout=5),
        ):
            resp = client.post(
                "/run",
                data=json.dumps({"command": "whoami", "args": []}),
                content_type="application/json",
                headers=_auth_headers(),
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["exit_code"] == -1
        assert "timed out" in data["stderr"].lower()

    def test_file_not_found_returns_200_with_error(self, client) -> None:
        with patch(
            "rex.computers.agent_server.subprocess.run",
            side_effect=FileNotFoundError("No such file or directory"),
        ):
            resp = client.post(
                "/run",
                data=json.dumps({"command": "whoami", "args": []}),
                content_type="application/json",
                headers=_auth_headers(),
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["exit_code"] == -1
        assert "not found" in data["stderr"].lower()
