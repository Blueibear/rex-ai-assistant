"""CLI tests for ``rex mobile-api`` and ``rex mobile-user create``.

Matrix rows: FND-008, USR-007..USR-010.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.mobile_api.conftest import TEST_JWT_SECRET


@pytest.fixture()
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("REX_DATA_DIR", str(data_dir))
    monkeypatch.setenv("REX_JWT_SECRET", TEST_JWT_SECRET)
    return data_dir


@pytest.fixture()
def profile_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list:
    """Redirect profile creation into a temp dir; record calls."""
    import rex.identity as rex_identity

    calls: list = []
    original = rex_identity.create_user_profile

    def _sandboxed(user_id: str, name: str, **kwargs):
        calls.append({"user_id": user_id, "name": name})
        return original(user_id, name, memory_dir=tmp_path / "Memory", **kwargs)

    monkeypatch.setattr(rex_identity, "create_user_profile", _sandboxed)
    return calls


def _user_rows(data_dir: Path) -> list:
    db = data_dir / "users.db"
    if not db.exists():
        return []
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute("SELECT username FROM users").fetchall()
    finally:
        conn.close()


class _FakeApp:
    def __init__(self, tls_material=None) -> None:
        self.extensions: dict = {"mobile_api_tls": tls_material}
        self.run_kwargs: dict | None = None

    def run(self, **kwargs) -> None:
        self.run_kwargs = kwargs


class _FakeTlsMaterial:
    def __init__(self, fingerprint: str = "aa" * 32) -> None:
        self.fingerprint_sha256 = fingerprint
        self.ssl_context_built = False

    def build_ssl_context(self):
        self.ssl_context_built = True
        return "fake-ssl-context"


class TestMobileApiCommand:
    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        cli_host: str | None,
        cli_port: int | None,
        config_host: str = "127.0.0.1",
        config_port: int = 8765,
        tls_material=None,
        create_error: BaseException | None = None,
    ) -> tuple[dict, _FakeApp | None]:
        """Run cmd_mobile_api with fakes; return (captured kwargs, fake app)."""
        import rex.config as rex_config
        import rex.mobile_api.app as mobile_app
        from rex.commands.mobile import cmd_mobile_api
        from rex.config import MobileApiConfig

        captured: dict = {}
        holder: dict = {}

        def _fake_create(config=None, services=None):
            captured["config"] = config
            if create_error is not None:
                raise create_error
            app = _FakeApp(tls_material=tls_material)
            holder["app"] = app
            return app

        fake_settings = SimpleNamespace(
            mobile_api=MobileApiConfig(host=config_host, port=config_port)
        )
        monkeypatch.setattr(rex_config, "load_config", lambda: fake_settings)
        monkeypatch.setattr(mobile_app, "create_mobile_app", _fake_create)

        args = argparse.Namespace(host=cli_host, port=cli_port)
        result = cmd_mobile_api(args)
        captured["result"] = result
        return captured, holder.get("app")

    def test_flags_override_config(self, monkeypatch, capsys) -> None:
        # "0.0.0.0" here only exercises CLI flag precedence against a fake
        # app object — nothing binds a socket in this test.
        captured, app = self._run(
            monkeypatch,
            cli_host="0.0.0.0",  # nosec B104
            cli_port=9000,
            config_host="127.0.0.1",
            config_port=8765,
        )
        assert captured["result"] == 0
        assert app is not None
        assert app.run_kwargs == {"host": "0.0.0.0", "port": 9000}  # nosec B104
        output = capsys.readouterr().out
        assert "0.0.0.0:9000" in output

    def test_config_used_when_no_flags(self, monkeypatch, capsys) -> None:
        captured, app = self._run(
            monkeypatch,
            cli_host=None,
            cli_port=None,
            config_host="127.0.0.1",
            config_port=9100,
        )
        assert captured["result"] == 0
        assert app is not None
        assert app.run_kwargs == {"host": "127.0.0.1", "port": 9100}
        output = capsys.readouterr().out
        assert "/mobile/status" in output
        assert "TLS: disabled (loopback development bind)" in output

    def test_tls_enabled_bind_prints_fingerprint_and_passes_ssl_context(
        self, monkeypatch, capsys
    ) -> None:
        material = _FakeTlsMaterial(fingerprint="bb" * 32)
        captured, app = self._run(
            monkeypatch,
            cli_host="192.168.1.50",
            cli_port=8765,
            tls_material=material,
        )
        assert captured["result"] == 0
        assert app is not None
        assert app.run_kwargs is not None
        assert app.run_kwargs["host"] == "192.168.1.50"
        assert app.run_kwargs["ssl_context"] == "fake-ssl-context"
        assert material.ssl_context_built is True
        output = capsys.readouterr().out
        assert "https://192.168.1.50:8765/mobile/status" in output
        assert "TLS: enabled" in output
        assert "bb" * 32 in output

    def test_tls_provisioning_failure_fails_closed(self, monkeypatch, capsys) -> None:
        from rex.mobile_api.tls import MobileTlsConfigurationError

        captured, app = self._run(
            monkeypatch,
            cli_host="192.168.1.50",
            cli_port=8765,
            create_error=MobileTlsConfigurationError("cert material could not be provisioned"),
        )
        assert captured["result"] == 1
        assert app is None
        output = capsys.readouterr().out
        assert "cert material could not be provisioned" in output
        assert "Non-loopback binds require usable TLS" in output

    def test_invalid_port_fails_before_serving(self, monkeypatch, capsys) -> None:
        import rex.config as rex_config
        from rex.commands.mobile import cmd_mobile_api
        from rex.config import MobileApiConfig

        fake_settings = SimpleNamespace(mobile_api=MobileApiConfig())
        monkeypatch.setattr(rex_config, "load_config", lambda: fake_settings)
        args = argparse.Namespace(host=None, port=99999)
        assert cmd_mobile_api(args) == 1

    def test_banner_never_prints_secret(self, monkeypatch, capsys) -> None:
        self._run(monkeypatch, cli_host=None, cli_port=None)
        assert TEST_JWT_SECRET not in capsys.readouterr().out


class TestMobileUserCreate:
    def _run_create(self, monkeypatch: pytest.MonkeyPatch, username: str, prompts: list) -> int:
        import getpass

        from rex.commands.mobile import cmd_mobile_user

        prompt_iter = iter(prompts)

        def _fake_getpass(prompt: str = "") -> str:
            value = next(prompt_iter)
            if isinstance(value, BaseException):
                raise value
            return value

        monkeypatch.setattr(getpass, "getpass", _fake_getpass)
        args = argparse.Namespace(mobile_user_command="create", username=username)
        return cmd_mobile_user(args)

    def test_creates_user_profile_and_first_admin(
        self, cli_env: Path, profile_sandbox: list, monkeypatch, capsys
    ) -> None:
        """USR-007: user, profile, and first-user admin created canonically."""
        from rex.permissions import get_permissions

        result = self._run_create(monkeypatch, "james", ["pw-123456", "pw-123456"])
        assert result == 0
        rows = _user_rows(cli_env)
        assert [row[0] for row in rows] == ["james"]
        assert len(profile_sandbox) == 1
        user_id = profile_sandbox[0]["user_id"]
        assert "admin" in get_permissions(user_id)
        output = capsys.readouterr().out
        assert "pw-123456" not in output  # USR-009: never echoed

    def test_second_user_is_not_admin(
        self, cli_env: Path, profile_sandbox: list, monkeypatch
    ) -> None:
        from rex.permissions import get_permissions

        assert self._run_create(monkeypatch, "james", ["pw-1", "pw-1"]) == 0
        assert self._run_create(monkeypatch, "sarah", ["pw-2", "pw-2"]) == 0
        assert "admin" not in get_permissions(profile_sandbox[1]["user_id"])

    def test_duplicate_username_fails_without_partial_state(
        self, cli_env: Path, profile_sandbox: list, monkeypatch
    ) -> None:
        """USR-008: safe nonzero failure; no extra profile or permissions."""
        assert self._run_create(monkeypatch, "james", ["pw-1", "pw-1"]) == 0
        assert self._run_create(monkeypatch, "james", ["pw-2", "pw-2"]) == 1
        assert len(_user_rows(cli_env)) == 1
        assert len(profile_sandbox) == 1

    def test_password_mismatch_creates_nothing(
        self, cli_env: Path, profile_sandbox: list, monkeypatch
    ) -> None:
        assert self._run_create(monkeypatch, "james", ["pw-1", "different"]) == 1
        assert _user_rows(cli_env) == []
        assert profile_sandbox == []

    def test_interrupted_prompt_creates_nothing(
        self, cli_env: Path, profile_sandbox: list, monkeypatch
    ) -> None:
        """USR-010: an interrupted prompt leaves no partial user record."""
        result = self._run_create(monkeypatch, "james", [KeyboardInterrupt()])
        assert result == 1
        assert _user_rows(cli_env) == []
        assert profile_sandbox == []

    def test_empty_password_rejected(
        self, cli_env: Path, profile_sandbox: list, monkeypatch
    ) -> None:
        assert self._run_create(monkeypatch, "james", ["", ""]) == 1
        assert _user_rows(cli_env) == []

    def test_cli_has_no_password_argument(self) -> None:
        """USR-009: the parser never accepts a password through argv."""
        from rex.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["mobile-user", "create", "--username", "a", "--password", "b"])
