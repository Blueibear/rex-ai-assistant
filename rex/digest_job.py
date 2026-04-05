"""Digest job for batching low-priority notifications (US-090).

Collects all medium and low priority notifications from the digest queue at a
configurable interval and delivers them as a single grouped message to the
dashboard notification store.

When no real delivery backend is configured the job logs the digest payload
instead (beta stub behaviour).

Usage example::

    from rex.digest_job import DigestJob
    from rex.priority_notification_router import PriorityNotificationRouter

    router = PriorityNotificationRouter()
    job = DigestJob(router=router)

    # Run manually (e.g. from a scheduler):
    result = job.run()
    print(result.entries_processed, result.delivered)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from rex.priority_notification_router import DigestEntry, PriorityNotificationRouter

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_MINUTES: int = 60


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DigestJobConfig:
    """Runtime configuration for :class:`DigestJob`.

    Attributes:
        interval_minutes: How often the digest job should run.
            Defaults to ``60``.  The job itself does not schedule itself;
            the caller is responsible for respecting this value.
    """

    interval_minutes: int = _DEFAULT_INTERVAL_MINUTES


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class DigestResult:
    """Outcome of a single :meth:`DigestJob.run` call.

    Attributes:
        ran_at: UTC timestamp when the job ran.
        entries_processed: Number of digest queue entries collected.
        delivered: ``True`` when the grouped message was written to the
            dashboard store or logged successfully.
        notification_id: ID assigned to the grouped notification (if any).
        error: Human-readable error string on failure.
    """

    ran_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    entries_processed: int = 0
    delivered: bool = False
    notification_id: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# DigestJob
# ---------------------------------------------------------------------------


class DigestJob:
    """Batches queued medium/low notifications into a single grouped message.

    Args:
        router: :class:`~rex.priority_notification_router.PriorityNotificationRouter`
            whose digest queue will be drained on each run.
        store: Optional store object to write the grouped message to.
            Must expose a ``write(notification_id, title, body, channel,
            metadata)`` method.  When ``None`` the job logs the digest
            payload (beta stub behaviour).
        config: Runtime configuration.  Defaults to
            :class:`DigestJobConfig` with a 60-minute interval.
    """

    def __init__(
        self,
        router: PriorityNotificationRouter,
        store: Any | None = None,
        config: DigestJobConfig | None = None,
    ) -> None:
        self._router = router
        self._store = store
        self._config = config or DigestJobConfig()
        self._last_run: datetime | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config(self) -> DigestJobConfig:
        """Current job configuration."""
        return self._config

    def configure(self, config: DigestJobConfig) -> None:
        """Replace config at runtime (no code changes or restarts required)."""
        self._config = config

    def configure_from_dict(self, data: dict[str, Any]) -> None:
        """Load and apply config from a plain dict.

        Args:
            data: Dict with optional ``"interval_minutes"`` key.
        """
        interval = int(data.get("interval_minutes", _DEFAULT_INTERVAL_MINUTES))
        self._config = DigestJobConfig(interval_minutes=interval)

    @property
    def last_run(self) -> datetime | None:
        """UTC timestamp of the most recent successful run, or ``None``."""
        return self._last_run

    @property
    def interval_minutes(self) -> int:
        """Configured digest interval in minutes."""
        return self._config.interval_minutes

    def run(self) -> DigestResult:
        """Drain the digest queue and deliver a single grouped notification.

        Steps:

        1. Drain all entries from the router's digest queue (clears queue).
        2. If the queue was empty, log and return without delivering.
        3. Build a grouped message summarising all collected entries.
        4. Write the grouped message to the dashboard store (or log if no
           store configured — beta stub behaviour).
        5. Record ``_last_run`` and return a :class:`DigestResult`.

        Returns:
            A :class:`DigestResult` describing the outcome.
        """
        ran_at = datetime.now(UTC)
        entries: list[DigestEntry] = self._router.drain_digest_queue()

        if not entries:
            logger.info(
                "[DigestJob] Digest run at %s: no queued notifications — skipping.",
                ran_at.isoformat(),
            )
            self._last_run = ran_at
            return DigestResult(ran_at=ran_at, entries_processed=0, delivered=False)

        title, body = self._build_digest_message(entries)
        nid = f"digest_{uuid.uuid4().hex[:16]}"

        try:
            self._deliver(nid=nid, title=title, body=body, entries=entries)
            self._last_run = ran_at
            logger.info(
                "[DigestJob] Digest delivered: id=%s entries=%d",
                nid,
                len(entries),
            )
            return DigestResult(
                ran_at=ran_at,
                entries_processed=len(entries),
                delivered=True,
                notification_id=nid,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[DigestJob] Delivery failed: %s", exc)
            self._last_run = ran_at
            return DigestResult(
                ran_at=ran_at,
                entries_processed=len(entries),
                delivered=False,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_digest_message(entries: list[DigestEntry]) -> tuple[str, str]:
        """Build a human-readable grouped digest title and body."""
        count = len(entries)
        title = f"Notification digest ({count} item{'s' if count != 1 else ''})"
        lines: list[str] = []
        for i, entry in enumerate(entries, start=1):
            notif = entry.notification
            lines.append(
                f"{i}. [{notif.priority.value.upper()}] {notif.title}"
                + (f" — {notif.body}" if notif.body else "")
            )
        body = "\n".join(lines)
        return title, body

    def _deliver(
        self,
        *,
        nid: str,
        title: str,
        body: str,
        entries: list[DigestEntry],
    ) -> None:
        """Write to dashboard store or log (stub) if no store configured."""
        if self._store is not None:
            self._store.write(
                notification_id=nid,
                title=title,
                body=body,
                channel="dashboard",
                metadata={"digest": True, "entry_count": len(entries)},
            )
        else:
            # Beta stub behaviour: log instead of real delivery.
            logger.info(
                "[DigestJob] STUB delivery (no backend configured):\n"
                "  id=%s\n  title=%s\n  body=%s",
                nid,
                title,
                body,
            )


__all__ = [
    "DigestJob",
    "DigestJobConfig",
    "DigestResult",
]
