"""DigestBuilder: batch low-priority notifications into a single summary.

:class:`DigestBuilder` collects all unread ``digest_eligible`` notifications
from :class:`~rex.notifications.models.NotificationStore`, asks an LLM to
summarise them in one paragraph, then dispatches a single desktop notification
with the summary and marks the source notifications as delivered.

If there are no digest-eligible notifications :meth:`DigestBuilder.build_digest`
returns ``None`` and :meth:`DigestBuilder.run_digest` is a no-op.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from rex.notifications.models import Notification, NotificationStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DigestBackend(Protocol):
    """Structural protocol for any object that can generate text."""

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
    ) -> str:
        """Return a generated text response."""
        ...


# ---------------------------------------------------------------------------
# Desktop notification helper
# ---------------------------------------------------------------------------


def _send_desktop(title: str, body: str) -> None:  # pragma: no cover
    """Send an OS-level desktop notification via plyer (logs on failure)."""
    try:
        from plyer import notification  # type: ignore[import-not-found]

        notification.notify(title=title, message=body, app_name="Rex")
    except Exception as exc:  # noqa: BLE001
        logger.info("Desktop notification (plyer unavailable): %s — %s | %s", title, body, exc)


# ---------------------------------------------------------------------------
# DigestBuilder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are Rex, a personal AI assistant. "
    "Produce a concise, one-paragraph digest summary of the pending updates "
    "listed by the user. Start with 'You have {count} update(s):'. "
    "Keep the summary under 120 words."
)

_USER_TEMPLATE = (
    "Please summarise these {count} low-priority notifications into one paragraph:\n\n" "{items}"
)


def _build_items_text(notifications: list[Notification]) -> str:
    lines: list[str] = []
    for i, n in enumerate(notifications, 1):
        lines.append(f"{i}. [{n.source}] {n.title}: {n.body}")
    return "\n".join(lines)


class DigestBuilder:
    """Build and dispatch a batched digest of low-priority notifications.

    Args:
        store: :class:`~rex.notifications.models.NotificationStore` to read
            from and update.
        backend: LLM backend implementing :class:`DigestBackend`.  When
            ``None`` a simple fallback summary is generated without an LLM.
    """

    def __init__(
        self,
        store: NotificationStore,
        backend: DigestBackend | None = None,
    ) -> None:
        self._store = store
        self._backend = backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_digest(self) -> str | None:
        """Generate a digest summary string, or ``None`` if nothing to digest.

        Collects all unread ``digest_eligible`` notifications.  If none
        exist returns ``None``.  Otherwise calls the LLM (or a fallback)
        to produce a one-paragraph summary and returns it.
        """
        eligible = [n for n in self._store.get_unread() if n.digest_eligible]
        if not eligible:
            return None

        count = len(eligible)
        items_text = _build_items_text(eligible)

        if self._backend is not None:
            try:
                summary = self._backend.generate(
                    [
                        {"role": "system", "content": _SYSTEM_PROMPT.format(count=count)},
                        {
                            "role": "user",
                            "content": _USER_TEMPLATE.format(count=count, items=items_text),
                        },
                    ],
                    max_tokens=256,
                )
                return summary.strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("DigestBuilder LLM call failed, using fallback: %s", exc)

        # Fallback: plain enumeration
        return f"You have {count} update(s): " + "; ".join(
            f"[{n.source}] {n.title}" for n in eligible
        )

    def run_digest(self) -> None:
        """Build digest, dispatch one desktop notification, mark as delivered.

        If :meth:`build_digest` returns ``None`` this method is a no-op.
        After dispatching the desktop notification all source notifications
        that were ``digest_eligible`` and unread are marked as read.
        """
        eligible = [n for n in self._store.get_unread() if n.digest_eligible]
        if not eligible:
            return

        summary = self.build_digest()
        if summary is None:
            return

        _send_desktop("Rex Digest", summary)

        now = datetime.now(timezone.utc)
        for n in eligible:
            delivered = n.model_copy(update={"delivered_at": now})
            self._store.update(delivered)
            self._store.mark_read(n.id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DigestBackend",
    "DigestBuilder",
]
