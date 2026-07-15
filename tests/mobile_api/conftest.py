"""Shared fixtures for mobile API gateway tests.

All tests use temporary data directories, an injected controllable clock,
and deterministic ID generation.  No test touches the real ``data/`` or
``Memory/`` directories or leaves repository changes behind.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

TEST_JWT_SECRET = (
    "unit-test-jwt-secret-0123456789abcdef0123456789abcdef"  # pragma: allowlist secret
)


class FakeClock:
    """Controllable UTC clock.

    Starts at the real current time so PyJWT's real-time ``exp``/``nbf``
    validation agrees with tokens issued through the injected clock, then
    advances only when a test says so.
    """

    def __init__(self, start: datetime | None = None) -> None:
        self.current = start or datetime.now(UTC)

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float = 0, days: float = 0) -> None:
        self.current += timedelta(seconds=seconds, days=days)


def sequential_token_generator(prefix: str = "refresh") -> Callable[[], str]:
    """Deterministic high-length token generator for storage tests."""
    counter = itertools.count(1)

    def _generate() -> str:
        return f"{prefix}-token-{next(counter):04d}-" + "x" * 43

    return _generate


@pytest.fixture()
def mobile_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the canonical stores at a temp dir and set a strong JWT secret."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("REX_DATA_DIR", str(data_dir))
    monkeypatch.setenv("REX_JWT_SECRET", TEST_JWT_SECRET)
    return data_dir


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def mobile_config():
    from rex.config import MobileApiConfig

    return MobileApiConfig()


class RecordingAuditLogger:
    """In-memory stand-in for rex.audit.AuditLogger — writes no files."""

    def __init__(self) -> None:
        self.entries: list = []

    def log(self, entry) -> None:
        self.entries.append(entry)


@pytest.fixture()
def audit_recorder() -> RecordingAuditLogger:
    return RecordingAuditLogger()


@pytest.fixture()
def services(
    mobile_env: Path,
    clock: FakeClock,
    mobile_config,
    audit_recorder: RecordingAuditLogger,
):
    from rex.mobile_api.db import migrate_users_db
    from rex.mobile_api.services import MobileApiServices

    db_path = mobile_env / "users.db"
    migrate_users_db(db_path)
    return MobileApiServices.build(
        mobile_config, db_path=db_path, clock=clock, audit_logger=audit_recorder
    )


@pytest.fixture()
def app(services):
    from rex.mobile_api.app import create_mobile_app

    application = create_mobile_app(services=services)
    application.config["TESTING"] = True
    return application


@pytest.fixture()
def client(app):
    with app.test_client() as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_user(username: str, password: str, *, admin: bool = False) -> str:
    """Create a canonical user (env-scoped tmp db) and return its ID."""
    from rex.auth import create_user as _create
    from rex.permissions import grant_permission

    user = _create(username, password)
    if admin:
        grant_permission(user["id"], "admin")
    return str(user["id"])


def disable_user(db_path: Path, user_id: str) -> None:
    from rex.mobile_api.db import connect

    conn = connect(db_path)
    try:
        conn.execute(
            "UPDATE users SET disabled_at = ? WHERE id = ?",
            (datetime.now(UTC).isoformat(), user_id),
        )
    finally:
        conn.close()


def login(client, username: str, password: str, device: dict | None = None):
    payload: dict = {"username": username, "password": password}
    if device is not None:
        payload["device"] = device
    return client.post("/mobile/auth/login", json=payload)


def auth_header(access_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token}"}


def login_tokens(client, username: str, password: str) -> dict:
    response = login(client, username, password)
    assert response.status_code == 200, response.get_json()
    return response.get_json()
