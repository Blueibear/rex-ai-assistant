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
from importlib.util import find_spec
from pathlib import Path
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

    def availability(self) -> tuple[bool, str]:
        """Check configured LLM prerequisites without loading a model or making a request."""
        try:
            from rex.config import settings  # noqa: PLC0415

            provider = str(settings.llm_provider).strip().lower()
            model = str(settings.llm_model or "").strip()
            if not model:
                return False, "no LLM model configured"
            if provider == "openai":
                ready = (
                    find_spec("openai") is not None
                    and bool(settings.openai_api_key)
                    and bool(settings.openai_model)
                )
                return (ready, "ok" if ready else "OpenAI is not fully configured")
            if provider == "anthropic":
                ready = (
                    find_spec("anthropic") is not None
                    and bool(settings.anthropic_api_key)
                    and bool(settings.anthropic_model)
                )
                return (ready, "ok" if ready else "Anthropic is not fully configured")
            if provider == "ollama":
                # A configured URL/package cannot prove that the requested local model exists.
                return False, "Ollama model readiness is not locally proven"
            if provider == "transformers":
                if find_spec("torch") is None or find_spec("transformers") is None:
                    return False, "transformers runtime is not installed"
                path = Path(model)
                if path.exists():
                    return True, "ok"
                try:
                    from transformers.utils.hub import try_to_load_from_cache  # noqa: PLC0415

                    cached = try_to_load_from_cache(model, "config.json")
                    ready = isinstance(cached, str) and Path(cached).is_file()
                    return (ready, "ok" if ready else "transformers model is not cached")
                except Exception:
                    return False, "transformers model readiness check failed"
            if provider == "echo":
                return True, "ok"
            return False, f"unsupported LLM provider '{provider}'"
        except Exception:
            return False, "LLM configuration is unavailable"

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

    def generate(
        self,
        message: str,
        *,
        user_id: str,
        voice_mode: bool = False,
        capability_scopes: frozenset[str] | None = None,
        capability_permissions: frozenset[str] | None = None,
        authorization_check: Callable[[], None] | None = None,
    ) -> str:
        """Return one complete reply under the server-derived mobile grant."""
        from rex.mobile_api.action_context import (  # noqa: PLC0415
            MobileActionDeniedError,
            mobile_action_context,
        )

        assistant = self._get_assistant()
        try:
            with mobile_action_context(
                capability_scopes or frozenset(),
                permissions=capability_permissions or frozenset(),
                revalidate=authorization_check,
            ):
                return str(
                    asyncio.run(
                        assistant.generate_reply(
                            message,
                            voice_mode=voice_mode,
                            active_user_id=user_id,
                        )
                    )
                )
        except MobileApiError:
            raise
        except MobileActionDeniedError as exc:
            raise MobileApiError(
                "FORBIDDEN",
                "This paired device is not authorized for the requested action.",
                403,
            ) from exc
        except Exception as exc:
            logger.error("Mobile chat generate_reply failed: %s", type(exc).__name__)
            raise _backend_unavailable(exc) from exc

    def stream(
        self,
        message: str,
        *,
        user_id: str,
        capability_scopes: frozenset[str] | None = None,
        capability_permissions: frozenset[str] | None = None,
        authorization_check: Callable[[], None] | None = None,
    ) -> Iterator[str]:
        """Yield reply chunks for ``user_id`` via ``Assistant.stream_reply()``.

        Bridges the async generator onto a private event loop owned by the
        calling (request) thread.  Any runtime failure surfaces as a
        structured ``BACKEND_UNAVAILABLE`` error for the transport to emit
        as a terminal error event.
        """
        from rex.mobile_api.action_context import (  # noqa: PLC0415
            MobileActionDeniedError,
            mobile_action_context,
        )

        assistant = self._get_assistant()
        loop = asyncio.new_event_loop()
        agen = None
        try:
            with mobile_action_context(
                capability_scopes or frozenset(),
                permissions=capability_permissions or frozenset(),
                revalidate=authorization_check,
            ):
                agen = assistant.stream_reply(message, active_user_id=user_id)
                while True:
                    try:
                        chunk = loop.run_until_complete(agen.__anext__())
                    except StopAsyncIteration:
                        break
                    except MobileApiError:
                        raise
                    except MobileActionDeniedError as exc:
                        raise MobileApiError(
                            "FORBIDDEN",
                            "This paired device is not authorized for the requested action.",
                            403,
                        ) from exc
                    except Exception as exc:
                        logger.error(
                            "Mobile chat stream_reply failed: %s",
                            type(exc).__name__,
                        )
                        raise _backend_unavailable(exc) from exc
                    if chunk:
                        yield str(chunk)
        finally:
            if agen is not None:
                try:
                    loop.run_until_complete(agen.aclose())
                except Exception:  # pragma: no cover - close-time cleanup
                    logger.debug("Mobile chat stream close failed")
            loop.close()


__all__ = ["STATUS_COMPLETED", "MobileChatService"]
