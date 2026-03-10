"""Stub SMS backend backed by a local JSON file.

This backend is the default for offline development and testing.  It writes
sent messages to a JSON file and reads inbound messages from the same file.
No external API calls are ever made.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rex.messaging_backends.base import InboundSms, SmsBackend, SmsSendResult

logger = logging.getLogger(__name__)

_DEFAULT_FIXTURE = Path("data/mock_sms.json")


class StubSmsBackend(SmsBackend):
    """JSON-fixture SMS backend for offline development and tests."""

    def __init__(
        self,
        fixture_path: Path | None = None,
        default_from: str = "+15555551234",
    ) -> None:
        self._fixture_path = fixture_path or _DEFAULT_FIXTURE
        self._default_from = default_from
        self._ensure_fixture()

    # ------------------------------------------------------------------
    # SmsBackend interface
    # ------------------------------------------------------------------

    def send_sms(
        self,
        *,
        to: str,
        body: str,
        from_number: str | None = None,
    ) -> SmsSendResult:
        from_num = from_number or self._default_from
        sid = f"stub_{uuid.uuid4().hex[:16]}"
        record = {
            "sid": sid,
            "from": from_num,
            "to": to,
            "body": body,
            "direction": "outbound",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._append_record(record)
        logger.info("[STUB] Would send SMS to %s: %s", to, body[:50])
        return SmsSendResult(ok=True, message_sid=sid)

    def fetch_recent_inbound(self, *, limit: int = 20) -> list[InboundSms]:
        records = self._load_records()
        inbound = [r for r in records if r.get("direction") == "inbound"]
        inbound.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        results: list[InboundSms] = []
        for r in inbound[:limit]:
            ts_raw = r.get("timestamp")
            ts = _parse_dt(ts_raw) or datetime.now(timezone.utc)
            results.append(
                InboundSms(
                    sid=r.get("sid", ""),
                    from_number=r.get("from", ""),
                    to_number=r.get("to", ""),
                    body=r.get("body", ""),
                    received_at=ts,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def inject_inbound(
        self,
        *,
        from_number: str,
        to_number: str | None = None,
        body: str,
    ) -> None:
        """Inject an inbound message for testing."""
        record = {
            "sid": f"stub_{uuid.uuid4().hex[:16]}",
            "from": from_number,
            "to": to_number or self._default_from,
            "body": body,
            "direction": "inbound",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._append_record(record)

    @property
    def sent_messages(self) -> list[dict[str, Any]]:
        """Return outbound messages for test assertions."""
        records = self._load_records()
        return [r for r in records if r.get("direction") == "outbound"]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_fixture(self) -> None:
        self._fixture_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._fixture_path.exists():
            self._fixture_path.write_text(json.dumps({"messages": []}, indent=2), encoding="utf-8")

    def _load_records(self) -> list[dict[str, Any]]:
        try:
            raw = self._fixture_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                return data.get("messages", [])  # type: ignore[no-any-return]
            return []
        except Exception as exc:
            logger.warning("Failed to load stub SMS records: %s", exc)
            return []

    def _append_record(self, record: dict[str, Any]) -> None:
        records = self._load_records()
        records.append(record)
        try:
            self._fixture_path.write_text(
                json.dumps({"messages": records}, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("Failed to save stub SMS record: %s", exc)


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


__all__ = ["StubSmsBackend"]
