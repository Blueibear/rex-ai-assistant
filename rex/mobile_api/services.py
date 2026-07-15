"""Process-level service container for the mobile API gateway.

Holds neutral, reusable dependencies only.  User-private state is never
cached here — every private operation receives a validated user ID
explicitly.  Clock, token, and ID generators are injectable so tests are
deterministic, and the runtime adapters (Assistant chat service, STT, TTS,
idempotency store) are injectable so tests never touch heavy ML
dependencies.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rex.config import MobileApiConfig
from rex.mobile_api.auth import load_jwt_secret
from rex.mobile_api.chat import MobileChatService
from rex.mobile_api.db import default_users_db_path
from rex.mobile_api.idempotency import MobileMessageStore
from rex.mobile_api.sessions import MobileSessionStore
from rex.mobile_api.voice import SpeechToTextAdapter, TextToSpeechAdapter


def _default_id_generator() -> str:
    return str(uuid.uuid4())


@dataclass
class MobileApiServices:
    """Neutral dependencies shared by all mobile requests."""

    config: MobileApiConfig
    db_path: Path
    jwt_secret: str
    session_store: MobileSessionStore
    message_store: MobileMessageStore
    chat_service: MobileChatService
    stt: SpeechToTextAdapter
    tts: TextToSpeechAdapter
    id_generator: Callable[[], str] = field(default=_default_id_generator)

    @property
    def clock(self) -> Callable[[], datetime]:
        """Return the store's injected clock (single time source)."""
        return self.session_store.now

    @classmethod
    def build(
        cls,
        config: MobileApiConfig | None = None,
        *,
        db_path: Path | str | None = None,
        jwt_secret: str | None = None,
        clock: Callable[[], datetime] | None = None,
        token_generator: Callable[[], str] | None = None,
        id_generator: Callable[[], str] | None = None,
        audit_logger: object | None = None,
        message_store: MobileMessageStore | None = None,
        chat_service: MobileChatService | None = None,
        stt: SpeechToTextAdapter | None = None,
        tts: TextToSpeechAdapter | None = None,
    ) -> MobileApiServices:
        """Build the default production container with optional test overrides."""
        cfg = config or MobileApiConfig()
        resolved_db_path = Path(db_path) if db_path is not None else default_users_db_path()
        secret = jwt_secret if jwt_secret is not None else load_jwt_secret()
        store = MobileSessionStore(
            resolved_db_path,
            refresh_ttl_seconds=cfg.refresh_token_ttl_days * 86400,
            clock=clock,
            token_generator=token_generator,
            id_generator=id_generator,
            audit_logger=audit_logger,
        )
        messages = message_store or MobileMessageStore(
            resolved_db_path,
            retention_hours=cfg.idempotency_retention_hours,
            clock=clock,
        )
        return cls(
            config=cfg,
            db_path=resolved_db_path,
            jwt_secret=secret,
            session_store=store,
            message_store=messages,
            chat_service=chat_service or MobileChatService(),
            stt=stt or SpeechToTextAdapter(),
            tts=tts or TextToSpeechAdapter(),
            id_generator=id_generator or _default_id_generator,
        )


__all__ = ["MobileApiServices"]
