"""Canonical Assistant adapter for the mobile gateway (issue #323, Session 2).

``MobileChatService`` is the only place mobile transports touch the Rex
runtime.  It:

- lazily builds one process-level *unbound* ``Assistant`` (issue #303: no
  implicit identity; construction performs no user-scoped reads/writes);
- passes the validated request identity explicitly as ``active_user_id`` to
  ``Assistant.generate_reply()`` / ``Assistant.stream_reply()`` on every
  call — it never mutates a shared current user;
- never calls ``LanguageModel`` directly (the GUI direct-LLM helper is an
  explicitly documented anti-pattern for this gateway);
- translates runtime failures into truthful structured errors
  (``BACKEND_UNAVAILABLE``) — never a mock reply.

Ordinary conversational output is ``completed``; ``verified`` is reserved
for real completion evidence, which this adapter never fabricates.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable, Iterator
from typing import Any

from rex.mobile_api import errors as merr
from rex.mobile_api.errors import MobileApiError

logger = logging.getLogger(__name__)

# Canonical completion status for a normal conversational reply.
STATUS_COMPLETED = "completed"


def _default_assistant_factory() -> Any:
    """Build the canonical unbound Assistant (no implicit identity)."""
    from rex.assistant import Assistant  # noqa: PLC0415 - heavy import on demand

    return Assistant()


def _backend_unavailable(exc: Exception) -> MobileApiError:
    # The client receives a generic truthful failure; details go to the
    # server log only (never a mock reply, never internal text on the wire).
    return MobileApiError(
        merr.BACKEND_UNAVAILABLE,
        "Rex is temporarily unavailable.",
        503,
        retryable=True,
    )


class MobileChatService:
    """Thread-safe adapter from mobile transports to the canonical Assistant."""

    def __init__(self, assistant_factory: Callable[[], Any] | None = None) -> None:
        self._assistant_factory = assistant_factory or _default_assistant_factory
        self._assistant: Any = None
        self._init_lock = threading.Lock()

    def _get_assistant(self) -> Any:
        if self._assistant is not None:
            return self._assistant
        with self._init_lock:
            if self._assistant is None:
                try:
                    self._assistant = self._assistant_factory()
                except Exception as exc:
                    logger.error("Mobile chat Assistant initialization failed: %s", exc)
                    raise _backend_unavailable(exc) from exc
        return self._assistant

    def generate(self, message: str, *, user_id: str, voice_mode: bool = False) -> str:
        """Return one complete reply for ``user_id`` via the canonical Assistant."""
        assistant = self._get_assistant()
        try:
            return str(
                asyncio.run(
                    assistant.generate_reply(message, voice_mode=voice_mode, active_user_id=user_id)
                )
            )
        except MobileApiError:
            raise
        except Exception as exc:
            logger.error("Mobile chat generate_reply failed: %s", exc)
            raise _backend_unavailable(exc) from exc

    def stream(self, message: str, *, user_id: str) -> Iterator[str]:
        """Yield reply chunks for ``user_id`` via ``Assistant.stream_reply()``.

        Bridges the async generator onto a private event loop owned by the
        calling (request) thread.  Any runtime failure surfaces as a
        structured ``BACKEND_UNAVAILABLE`` error for the transport to emit
        as a terminal error event.
        """
        assistant = self._get_assistant()
        loop = asyncio.new_event_loop()
        agen = None
        try:
            agen = assistant.stream_reply(message, active_user_id=user_id)
            while True:
                try:
                    chunk = loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    break
                except MobileApiError:
                    raise
                except Exception as exc:
                    logger.error("Mobile chat stream_reply failed: %s", exc)
                    raise _backend_unavailable(exc) from exc
                if chunk:
                    yield str(chunk)
        finally:
            if agen is not None:
                try:
                    loop.run_until_complete(agen.aclose())
                except Exception:  # pragma: no cover - close-time cleanup
                    logger.debug("Mobile chat stream close failed", exc_info=True)
            loop.close()


__all__ = ["STATUS_COMPLETED", "MobileChatService"]
