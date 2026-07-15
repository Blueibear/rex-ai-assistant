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


# ---------------------------------------------------------------------------
# Fake runtime adapters (Session 2) — deterministic, no ML dependencies.
# ---------------------------------------------------------------------------


class FakeChatService:
    """Deterministic stand-in for MobileChatService.

    Replies embed the calling user's ID so cross-user isolation failures are
    visible, and every execution is recorded so idempotency tests can assert
    exactly-once behaviour.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail_with: Exception | None = None
        self.stream_fail_after_chunks: int | None = None

    def _reply(self, message: str, user_id: str) -> str:
        return f"echo[{user_id}]: {message}"

    def generate(self, message: str, *, user_id: str, voice_mode: bool = False) -> str:
        if self.fail_with is not None:
            raise self.fail_with
        self.calls.append((message, user_id))
        return self._reply(message, user_id)

    def stream(self, message: str, *, user_id: str):
        if self.fail_with is not None:
            raise self.fail_with
        self.calls.append((message, user_id))
        text = self._reply(message, user_id)
        chunks = [text[i : i + 6] for i in range(0, len(text), 6)]
        for index, chunk in enumerate(chunks):
            if self.stream_fail_after_chunks is not None and index >= self.stream_fail_after_chunks:
                from rex.mobile_api import errors as merr
                from rex.mobile_api.errors import MobileApiError

                raise MobileApiError(
                    merr.BACKEND_UNAVAILABLE,
                    "Rex is temporarily unavailable.",
                    503,
                    retryable=True,
                )
            yield chunk


class FakeSttAdapter:
    """Deterministic stand-in for SpeechToTextAdapter."""

    def __init__(self) -> None:
        self.available = True
        self.unavailable_reason = "fake STT disabled"
        self.transcript = "turn off the downstairs lights"
        self.decode_seconds = 2.0
        self.decoded_paths: list[str] = []
        self.decode_fails = False

    def availability(self) -> tuple[bool, str]:
        return (True, "ok") if self.available else (False, self.unavailable_reason)

    def require_available(self) -> None:
        if not self.available:
            from rex.mobile_api import errors as merr
            from rex.mobile_api.errors import MobileApiError

            raise MobileApiError(
                merr.BACKEND_UNAVAILABLE,
                "Speech-to-text is not available on this server.",
                503,
                retryable=False,
            )

    def decode(self, path: str):
        self.require_available()
        self.decoded_paths.append(path)
        if self.decode_fails:
            from rex.mobile_api import errors as merr
            from rex.mobile_api.errors import MobileApiError

            raise MobileApiError(merr.INVALID_MEDIA, "The audio could not be decoded.", 415)
        from rex.mobile_api.voice import WHISPER_SAMPLE_RATE

        return [0.0] * int(self.decode_seconds * WHISPER_SAMPLE_RATE)

    def transcribe(self, audio) -> str:
        self.require_available()
        return self.transcript


class FakeTtsAdapter:
    """Deterministic stand-in for TextToSpeechAdapter."""

    def __init__(self) -> None:
        self.available = True
        self.known_voices = ["fake-default-voice", "fake-alt-voice"]
        self.audio = b"FAKE-TTS-AUDIO"
        self.synthesized: list[tuple[str, str]] = []

    def availability(self) -> tuple[bool, str]:
        return (True, "ok") if self.available else (False, "fake TTS disabled")

    def require_available(self) -> None:
        if not self.available:
            from rex.mobile_api import errors as merr
            from rex.mobile_api.errors import MobileApiError

            raise MobileApiError(
                merr.BACKEND_UNAVAILABLE,
                "Text-to-speech is not available on this server.",
                503,
                retryable=False,
            )

    def mime_type(self) -> str:
        return "audio/wav"

    def resolve_voice(self, requested: str | None) -> str:
        self.require_available()
        if requested is None or requested.strip() in ("", "default"):
            return self.known_voices[0]
        requested = requested.strip()
        if requested not in self.known_voices:
            from rex.mobile_api import errors as merr
            from rex.mobile_api.errors import MobileApiError

            raise MobileApiError(merr.BAD_REQUEST, "The requested voice is not available.", 400)
        return requested

    def synthesize(self, text: str, voice_id: str) -> bytes:
        self.require_available()
        self.synthesized.append((text, voice_id))
        return self.audio


@pytest.fixture()
def fake_chat_service() -> FakeChatService:
    return FakeChatService()


@pytest.fixture()
def fake_stt() -> FakeSttAdapter:
    return FakeSttAdapter()


@pytest.fixture()
def fake_tts() -> FakeTtsAdapter:
    return FakeTtsAdapter()


@pytest.fixture()
def services(
    mobile_env: Path,
    clock: FakeClock,
    mobile_config,
    audit_recorder: RecordingAuditLogger,
    fake_chat_service: FakeChatService,
    fake_stt: FakeSttAdapter,
    fake_tts: FakeTtsAdapter,
):
    from rex.mobile_api.db import migrate_users_db
    from rex.mobile_api.services import MobileApiServices

    db_path = mobile_env / "users.db"
    migrate_users_db(db_path)
    return MobileApiServices.build(
        mobile_config,
        db_path=db_path,
        clock=clock,
        audit_logger=audit_recorder,
        chat_service=fake_chat_service,
        stt=fake_stt,
        tts=fake_tts,
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


def chat_payload(message: str = "Hello Rex", **overrides) -> dict:
    """Canonical valid chat request body with fresh UUIDs."""
    import uuid

    payload = {
        "message_id": str(uuid.uuid4()),
        "conversation_id": str(uuid.uuid4()),
        "sent_at": "2026-07-15T12:00:00+00:00",
        "message": message,
        "mode": "mobile_text",
        "client_context": {"device": "iphone", "response_preference": "brief"},
    }
    payload.update(overrides)
    return payload


def parse_sse_events(body: bytes) -> list[dict]:
    """Parse an SSE body into its JSON event payloads (strict)."""
    import json as _json

    events = []
    for line in body.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        assert line.startswith("data: "), f"non-SSE line in stream: {line!r}"
        events.append(_json.loads(line[len("data: ") :]))
    return events
