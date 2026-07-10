"""
Email service module for Rex AI Assistant.

Provides email triage functionality with classification and summarization.

Design goals:
- Works in stub/mock mode (no real IMAP/SMTP yet) by default.
- Supports both:
  1) "fetch_unread(...)" returning structured EmailSummary objects
  2) "triage_unread(...)" returning CLI-friendly dict summaries
- Optional EventBus publishing if rex.event_bus is available and provided.
- Backend-aware: accepts an optional ``EmailBackend`` for real IMAP/SMTP.
- Multi-account: resolves backends per validated user and account via
  ``rex.email_accounts.EmailAccountResolver`` (issue #303).
- ``send()`` method delegates to the requesting user's authorized backend.

Per-user isolation (issue #303):
- Every operation that reads or mutates email data requires an explicit,
  validated ``user_id``.  Missing, blank, malformed, or traversal-style
  identities fail closed (``EmailIdentityError``) before any account or
  credential lookup.
- Account selection is restricted to the requesting user's authorized
  accounts; explicit foreign or nonexistent accounts raise the generic
  ``EmailAccountAccessError`` (indistinguishable to the caller).
- Backends are cached per ``(user_id, account_id)``; a backend resolved for
  one user is never reused for another.
- Stub/mock mode keeps a separate mutable inbox per user so one user's
  mark-read never alters another user's view.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rex.email_accounts import (
    DEFAULT_PROFILE,
    EmailAccountResolver,
    require_user_id,
)

logger = logging.getLogger(__name__)

# Optional EventBus import for compatibility with earlier implementation
try:
    from rex.openclaw.event_bus import EventBus
except Exception:  # pragma: no cover
    EventBus = Any  # type: ignore

# Optional credentials manager import
try:
    from rex.credentials import get_credential_manager
except Exception:  # pragma: no cover
    get_credential_manager = None  # type: ignore

try:
    from rex.assistant_errors import IntegrationNotConfiguredError
except Exception:  # pragma: no cover
    IntegrationNotConfiguredError = RuntimeError  # type: ignore

# Optional Pydantic import (nice to have, not required at runtime)
try:
    from pydantic import BaseModel, ConfigDict, Field, field_serializer

    class EmailSummary(BaseModel):
        """Summary of an email message."""

        model_config = ConfigDict()

        id: str = Field(..., description="Unique email identifier")
        from_addr: str = Field(..., description="Sender email address")
        subject: str = Field(..., description="Email subject")
        snippet: str = Field(..., description="Brief preview of email body")
        received_at: datetime = Field(..., description="When the email was received")
        labels: list[str] = Field(default_factory=list, description="Email labels/tags")
        importance_score: float = Field(default=0.5, description="Importance score (0.0-1.0)")
        category: str | None = Field(
            default=None, description="Email category (important, promo, social, etc.)"
        )

        @field_serializer("received_at", when_used="json")
        @classmethod
        def _serialize_received_at(cls, v: datetime, _info: object) -> str:
            return v.isoformat()

except Exception:  # pragma: no cover

    @dataclass
    class EmailSummary:  # type: ignore
        """Fallback EmailSummary when pydantic is not installed."""

        id: str
        from_addr: str
        subject: str
        snippet: str
        received_at: datetime
        labels: list[str]
        importance_score: float = 0.5
        category: str | None = None


@dataclass(frozen=True)
class EmailMessage:
    """Legacy/compat message type used by older CLI/service code."""

    message_id: str
    sender: str
    subject: str
    body: str
    received_at: datetime

    def to_summary(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "subject": self.subject,
            "received_at": self.received_at.isoformat(),
            "body": self.body,
        }


class EmailService:
    """
    Email service for reading, categorizing, and sending emails.

    Supports two modes:
    - **Stub mode** (default): reads from a local JSON fixture; send is a
      logged no-op.  Used for offline development and testing.  Each user
      gets an isolated mutable copy of the fixture inbox.
    - **Backend mode**: resolves an ``EmailBackend`` per validated user and
      authorized account via ``EmailAccountResolver``.  An explicitly
      injected backend (constructor ``backend=`` or :meth:`set_backend`)
      serves only the user it is bound to.

    Every read/mutate/send operation requires a validated ``user_id`` and
    fails closed on missing or invalid identity.
    """

    def __init__(
        self,
        mock_data_file: Path | None = None,
        *,
        event_bus: EventBus | None = None,
        mock_messages: list[EmailMessage] | None = None,
        mock_data_path: Path | None = None,
        backend: object | None = None,
        backend_user_id: str = DEFAULT_PROFILE,
        user_email_accounts: dict[str, list[Any]] | None = None,
        user_default_accounts: dict[str, str] | None = None,
        account_resolver: EmailAccountResolver | None = None,
    ) -> None:
        if event_bus is None and mock_data_file is not None and hasattr(mock_data_file, "publish"):
            event_bus = mock_data_file  # type: ignore[assignment]
            mock_data_file = None

        if mock_data_file is None and mock_data_path is not None:
            mock_data_file = mock_data_path

        self.mock_data_file = mock_data_file or Path("data/mock_emails.json")
        self.connected = False
        self._event_bus = event_bus
        self._mock_messages = mock_messages  # If provided, overrides file loading
        self._mock_emails: list[EmailSummary] = []
        # Per-user isolated copies of the stub inbox: {user_id: [EmailSummary]}
        self._user_mock_emails: dict[str, list[EmailSummary]] = {}
        # Explicitly injected backend, bound to exactly one user.
        self._backend = backend
        self._backend_owner = require_user_id(backend_user_id) if backend is not None else None
        # Per-user account map: {user_id: [UserEmailAccount, ...]}
        self._user_email_accounts: dict[str, list[Any]] = user_email_accounts or {}
        # Per-user default account selection: {user_id: account_id}
        self._user_default_accounts: dict[str, str] = user_default_accounts or {}
        # Injected authorization/routing resolver (tests/embedding only; the
        # production path loads lazily so config changes are picked up).
        self._account_resolver = account_resolver
        self._resolver_cache: EmailAccountResolver | None = None
        self._resolver_stamp: int | None = None
        # Backends resolved from config, cached per (user_id, account_id).
        # A backend resolved for one user is never served to another.
        self._user_backends: dict[tuple[str, str], object] = {}

        self.credential_manager = None
        if get_credential_manager is not None:
            try:
                self.credential_manager = get_credential_manager()
            except Exception:
                self.credential_manager = None

    # ------------------------------------------------------------------
    # Backend management
    # ------------------------------------------------------------------

    def set_backend(self, backend: object, *, user_id: str = DEFAULT_PROFILE) -> None:
        """Swap the explicitly injected email backend at runtime.

        The injected backend is bound to *user_id* and serves only that
        user's operations; other users resolve through the account resolver.

        Args:
            backend: An ``EmailBackend`` instance (or ``None`` for stub).
            user_id: The user this backend belongs to (default: the legacy
                ``default`` profile).
        """
        self._backend = backend
        self._backend_owner = require_user_id(user_id) if backend is not None else None
        self.connected = False

    @property
    def active_backend(self) -> object | None:
        """The explicitly injected backend (``None`` in stub/resolver mode)."""
        return self._backend

    # ------------------------------------------------------------------
    # User-scoped account access (US-ME-002 / issue #303)
    # ------------------------------------------------------------------

    def _get_resolver(self) -> EmailAccountResolver:
        """Return the account resolver for this service instance.

        Config-backed resolvers are invalidated when the runtime config file
        changes, so revoking or reassigning a user's accounts takes effect in
        long-lived processes without a restart.  Authorization is re-checked
        on every operation, so cached backends never outlive their owner's
        assignment.
        """
        if self._account_resolver is not None:
            return self._account_resolver

        from rex import email_accounts as _email_accounts

        stamp = _email_accounts.config_stamp()
        if self._resolver_cache is None or stamp != self._resolver_stamp:
            if self._user_email_accounts or self._user_default_accounts:
                # Explicitly injected authorization map: authoritative.
                base = EmailAccountResolver.load()
                self._resolver_cache = EmailAccountResolver(
                    base.email_config,
                    self._user_email_accounts,
                    self._user_default_accounts,
                )
            else:
                self._resolver_cache = EmailAccountResolver.load()
            self._resolver_stamp = stamp
        return self._resolver_cache

    def get_accounts(self, user_id: str) -> list[Any]:
        """Return the email account entries authorized for *user_id*.

        Returns an empty list (not an error) when the user has no accounts
        configured.  Never returns accounts belonging to other users.

        Raises:
            EmailIdentityError: On missing or invalid identity.
        """
        return self._get_resolver().entries_for_user(user_id)

    def _check_account_access(self, user_id: str, account_id: str) -> None:
        """Raise ``EmailAccountAccessError`` unless *user_id* owns *account_id*.

        Unauthorized and nonexistent accounts are indistinguishable.  An
        empty authorization map means "no access", never "allow all".
        """
        self._get_resolver().check_account_access(user_id, account_id)

    def _get_user_backend(
        self, user_id: str, account_id: str | None = None
    ) -> tuple[object | None, Any]:
        """Resolve the backend serving *user_id*, after ownership validation.

        Credential lookup happens only here, only after the account has been
        authorized, and only with that account definition's own
        ``credential_ref``.

        Returns:
            ``(backend_or_None, account_definition_or_None)``.
        """
        validated = require_user_id(user_id)
        resolver = self._get_resolver()
        resolved_id = resolver.resolve_account_id(validated, account_id)
        if resolved_id is None:
            return None, None
        definition = resolver.get_account_definition(resolved_id)
        if definition is None:
            return None, None

        key = (validated, resolved_id)
        backend = self._user_backends.get(key)
        if backend is None:
            credential_getter = None
            if self.credential_manager is not None and hasattr(
                self.credential_manager, "get_token"
            ):
                credential_getter = self.credential_manager.get_token

            from rex.email_backends.account_router import build_backend_for_account

            backend = build_backend_for_account(definition, credential_getter=credential_getter)
            if backend is None:
                return None, definition
            try:
                backend.connect()
            except Exception as exc:
                logger.warning(
                    "Failed to connect email backend for account %r: %s", resolved_id, exc
                )
            self._user_backends[key] = backend
        return backend, definition

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, user_id: str | None = None) -> bool:
        """
        Connect to email service.

        With an injected backend, delegates to ``backend.connect()``.  When
        real accounts are configured, *user_id* is required and only that
        user's authorized backend is connected.  Otherwise falls back to
        stub behaviour (loads mock data).
        """
        try:
            if self._backend is not None:
                result = self._backend.connect()  # type: ignore[attr-defined]
                self.connected = bool(result)
                if self.connected:
                    logger.info("Email service connected via backend")
                return self.connected

            resolver = self._get_resolver()
            if resolver.has_configured_accounts():
                if user_id is None:
                    logger.warning(
                        "Email accounts are configured; connect() requires a user identity"
                    )
                    return False
                backend, _definition = self._get_user_backend(user_id)
                if backend is None:
                    return False
                self.connected = True
                return True

            # Stub mode
            if self.credential_manager is not None:
                try:
                    email_creds = self.credential_manager.get_credential("email")
                    if not email_creds:
                        logger.warning("No email credentials configured (continuing in stub mode)")
                except Exception as e:
                    logger.warning("Credential manager error (continuing in stub mode): %s", e)

            self._load_mock_data()
            self.connected = True
            logger.info("Email service connected (stub mode)")
            return True
        except Exception as e:
            logger.error("Failed to connect email service: %s", e, exc_info=True)
            self.connected = False
            return False

    def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        if self._event_bus is None:
            return
        try:
            self._event_bus.publish(topic, payload)
        except Exception as e:
            logger.debug("EventBus publish failed for %s: %s", topic, e)

    def _publish_user_event(
        self,
        topic: str,
        user_id: str,
        shared_payload: dict[str, Any],
        private_payload: dict[str, Any] | None = None,
    ) -> None:
        """Publish a safe envelope on the shared topic and the full payload
        on the owner's user-scoped topic.

        The shared topic must never carry private email fields (subjects,
        senders, snippets); those go only to ``{topic}.user.{user_id}``.
        """
        self._publish(topic, shared_payload)
        if private_payload is not None:
            self._publish(f"{topic}.user.{user_id}", private_payload)

    def _load_mock_data(self) -> None:
        """Load mock email data from file or legacy mock messages."""
        if self._mock_messages is not None:
            self._mock_emails = [self._email_summary_from_message(m) for m in self._mock_messages]
            return

        if not self.mock_data_file.exists():
            logger.warning("No mock email data at %s", self.mock_data_file)
            self._mock_emails = []
            return

        try:
            raw = self.mock_data_file.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as e:
            logger.error("Failed to read mock email data: %s", e, exc_info=True)
            self._mock_emails = []
            return

        # Accept either:
        # 1) list[dict]
        # 2) {"messages": list[dict]}
        items: list[dict[str, Any]]
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and isinstance(data.get("messages"), list):
            items = [x for x in data["messages"] if isinstance(x, dict)]
        else:
            logger.warning(
                "Mock email data format not recognized, expected list or {messages: [...]}"
            )
            self._mock_emails = []
            return

        parsed: list[EmailSummary] = []
        for item in items:
            try:
                parsed.append(self._email_summary_from_dict(item))
            except Exception as e:
                logger.warning("Skipping invalid mock email record: %s", e)

        self._mock_emails = parsed
        logger.info("Loaded %d mock emails", len(self._mock_emails))

    def _user_inbox(self, user_id: str) -> list[EmailSummary]:
        """Return *user_id*'s isolated mutable copy of the stub inbox."""
        inbox = self._user_mock_emails.get(user_id)
        if inbox is None:
            inbox = copy.deepcopy(self._mock_emails)
            self._user_mock_emails[user_id] = inbox
        return inbox

    def _email_summary_from_message(self, msg: EmailMessage) -> EmailSummary:
        return EmailSummary(
            id=msg.message_id,
            from_addr=msg.sender,
            subject=msg.subject,
            snippet=(msg.body or "")[:200],
            received_at=msg.received_at,
            labels=["unread"],
            importance_score=0.5,
        )

    def _email_summary_from_dict(self, d: dict[str, Any]) -> EmailSummary:
        # Support both schema variants:
        # - legacy: message_id, sender, subject, body, received_at
        # - newer: id, from_addr, subject, snippet, received_at, labels, importance_score
        email_id = str(d.get("id") or d.get("message_id") or "")
        if not email_id:
            raise ValueError("missing id/message_id")

        from_addr = str(d.get("from_addr") or d.get("sender") or "unknown@example.com")
        subject = str(d.get("subject") or "")
        body = d.get("body")
        snippet = d.get("snippet")

        if snippet is None and body is not None:
            snippet = str(body)[:200]
        if snippet is None:
            snippet = ""

        received_at_raw = d.get("received_at")
        received_at = self._parse_datetime(received_at_raw) or datetime.now(UTC)

        labels = d.get("labels")
        if not isinstance(labels, list):
            labels = ["unread"]

        importance_score = d.get("importance_score")
        try:
            importance_score_f = float(importance_score) if importance_score is not None else 0.5
        except Exception:
            importance_score_f = 0.5

        category = d.get("category")
        category_str = str(category) if category is not None else None

        return EmailSummary(
            id=email_id,
            from_addr=from_addr,
            subject=subject,
            snippet=str(snippet),
            received_at=received_at,
            labels=[str(x) for x in labels],
            importance_score=importance_score_f,
            category=category_str,
        )

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
                return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def fetch_unread(
        self,
        limit: int = 10,
        *,
        user_id: str | None = None,
        account_id: str | None = None,
    ) -> list[EmailSummary]:
        """
        Fetch unread email summaries for the requesting user.

        Args:
            limit:      Maximum number of emails to return.
            user_id:    Requesting user (required).  Only that user's
                        authorized accounts are consulted.
            account_id: Optional explicit account, validated for ownership.

        Returns:
            List of EmailSummary objects.  Returns ``[]`` when the user has
            no authorized/usable account (documented not-configured result).

        Raises:
            EmailIdentityError: On missing or invalid identity.
            EmailAccountAccessError: When *account_id* is not available to
                this user (unauthorized or nonexistent).
        """
        validated = require_user_id(user_id)
        resolver = self._get_resolver()
        if account_id:
            resolver.check_account_access(validated, account_id)

        if self._backend is not None and self._backend_owner == validated:
            if not self.connected:
                if not self.connect():
                    logger.warning("Email service not connected")
                    return []
            envelopes = self._backend.fetch_unread(limit=limit)  # type: ignore[attr-defined]
            result = [self._envelope_to_summary(env) for env in envelopes]
            self._publish_user_event(
                "email.unread",
                validated,
                {"count": len(result), "user_id": validated, "account_id": account_id},
                {
                    "count": len(result),
                    "user_id": validated,
                    "account_id": account_id,
                    "messages": [self._summary_dict(e) for e in result],
                },
            )
            return result

        if resolver.has_configured_accounts():
            backend, definition = self._get_user_backend(validated, account_id)
            if backend is None:
                logger.warning("No usable email account for user %r", validated)
                return []
            resolved_id = getattr(definition, "id", None)
            envelopes = backend.fetch_unread(limit=limit)  # type: ignore[attr-defined]
            result = [self._envelope_to_summary(env) for env in envelopes]
            self._publish_user_event(
                "email.unread",
                validated,
                {"count": len(result), "user_id": validated, "account_id": resolved_id},
                {
                    "count": len(result),
                    "user_id": validated,
                    "account_id": resolved_id,
                    "messages": [self._summary_dict(e) for e in result],
                },
            )
            return result

        # Stub mode: per-user isolated inbox.
        if not self.connected:
            if not self.connect():
                logger.warning("Email service not connected")
                return []

        inbox = self._user_inbox(validated)
        unread = [email for email in inbox if "unread" in (email.labels or [])]
        result = unread[: max(0, int(limit))]
        self._publish_user_event(
            "email.unread",
            validated,
            {"count": len(result), "user_id": validated, "account_id": account_id},
            {
                "count": len(result),
                "user_id": validated,
                "account_id": account_id,
                "messages": [self._summary_dict(e) for e in result],
            },
        )
        return result

    def triage_unread(
        self,
        limit: int = 10,
        *,
        user_id: str | None = None,
        account_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Legacy-friendly triage output used by older CLI command.

        Requires a validated ``user_id``; triages only that user's unread
        mail.  Returns list of dicts:
            {
              message_id, sender, subject, received_at (datetime),
              category, summary
            }
        """
        validated = require_user_id(user_id)
        unread = self.fetch_unread(limit=limit, user_id=validated, account_id=account_id)
        triaged: list[dict[str, Any]] = []

        for email in unread:
            category = self.categorize(email)
            summary = self._triage_summary(email)
            triaged.append(
                {
                    "message_id": email.id,
                    "sender": email.from_addr,
                    "subject": email.subject,
                    "received_at": email.received_at,
                    "category": category,
                    "summary": summary,
                }
            )

        self._publish_user_event(
            "email.triaged",
            validated,
            {"count": len(triaged), "user_id": validated},
            {
                "count": len(triaged),
                "user_id": validated,
                "triaged": [
                    {"message_id": t["message_id"], "category": t["category"]} for t in triaged
                ],
            },
        )
        return triaged

    def mark_as_read(self, email_id: str, *, user_id: str | None = None) -> bool:
        """Mark an email as read within the requesting user's inbox."""
        validated = require_user_id(user_id)

        if self._backend is not None and self._backend_owner == validated:
            if not self.connected:
                logger.warning("Email service not connected")
                return False
            result = self._backend.mark_as_read(email_id)  # type: ignore[attr-defined]
            if result:
                self._publish("email.read", {"id": email_id, "user_id": validated})
            return result  # type: ignore[no-any-return]

        resolver = self._get_resolver()
        if resolver.has_configured_accounts():
            backend, _definition = self._get_user_backend(validated)
            if backend is None:
                logger.warning("No usable email account for user %r", validated)
                return False
            result = backend.mark_as_read(email_id)  # type: ignore[attr-defined]
            if result:
                self._publish("email.read", {"id": email_id, "user_id": validated})
            return bool(result)

        if not self.connected:
            logger.warning("Email service not connected")
            return False

        for email in self._user_inbox(validated):
            if email.id == email_id:
                if email.labels and "unread" in email.labels:
                    email.labels.remove("unread")
                logger.info("Marked email %s as read for user %s", email_id, validated)
                self._publish("email.read", {"id": email_id, "user_id": validated})
                return True

        logger.warning("Email not found: %s", email_id)
        return False

    # ------------------------------------------------------------------
    # Send operations
    # ------------------------------------------------------------------

    def send(
        self,
        *,
        to: str | list[str],
        subject: str,
        body: str,
        from_addr: str | None = None,
        account_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Send an email through the requesting user's authorized account.

        Args:
            to:         Recipient address(es).
            subject:    Email subject line.
            body:       Plain-text body.
            from_addr:  Sender address override (defaults to account address).
            account_id: Explicit account to route through, validated for
                        ownership.
            user_id:    Requesting user (required).

        Returns:
            A dict with keys ``ok`` (bool), ``message_id`` (str|None), and
            ``error`` (str|None).  When real accounts are configured but none
            is available to this user, ``ok`` is False (fail closed) — never
            a fallback through another user's account.  In pure stub mode
            (no accounts configured anywhere) the send is logged and reported
            as ok.

        Raises:
            EmailIdentityError: On missing or invalid identity.
            EmailAccountAccessError: When *account_id* is not available to
                this user (unauthorized or nonexistent).
        """
        validated = require_user_id(user_id)
        resolver = self._get_resolver()
        if account_id:
            resolver.check_account_access(validated, account_id)

        to_addrs = [to] if isinstance(to, str) else list(to)

        send_backend = None
        resolved_account = None
        if self._backend is not None and self._backend_owner == validated:
            send_backend = self._backend
        else:
            send_backend, resolved_account = self._get_user_backend(validated, account_id)

        if send_backend is not None and hasattr(send_backend, "send"):
            sender = from_addr or getattr(resolved_account, "address", "") or ""
            result = send_backend.send(
                from_addr=sender,
                to_addrs=to_addrs,
                subject=subject,
                body=body,
            )
            return {
                "ok": result.ok,
                "message_id": result.message_id,
                "error": result.error,
            }

        if resolver.has_configured_accounts():
            # Real accounts exist but none is available to this user (or its
            # credentials are missing): fail closed, never fall back.
            logger.warning("Email send refused: no authorized account for user %r", validated)
            return {
                "ok": False,
                "message_id": None,
                "error": "No email account is available for this user",
            }

        # Stub mode: log and return success.
        logger.info(
            "[STUB] Would send email to=%s subject=%r",
            to_addrs,
            subject,
        )
        return {"ok": True, "message_id": None, "error": None}

    # ------------------------------------------------------------------
    # Categorisation / summarisation
    # ------------------------------------------------------------------

    def categorize(self, email: EmailSummary) -> str:
        """
        Categorize an email based on heuristics.

        Returns:
            Category string: important, promo, social, newsletter, finance, calendar, general
        """
        subject_lower = (email.subject or "").lower()
        from_lower = (email.from_addr or "").lower()
        snippet_lower = (email.snippet or "").lower()

        # Finance indicators
        finance_keywords = ["invoice", "payment", "receipt", "billing", "charged", "refund"]
        if any(kw in subject_lower or kw in snippet_lower for kw in finance_keywords):
            return "finance"

        # Calendar indicators
        calendar_keywords = ["meeting", "schedule", "invite", "calendar", "appointment"]
        if any(kw in subject_lower or kw in snippet_lower for kw in calendar_keywords):
            return "calendar"

        # Promo indicators
        promo_keywords = [
            "sale",
            "discount",
            "offer",
            "deal",
            "promotion",
            "coupon",
            "free shipping",
        ]
        if any(kw in subject_lower or kw in snippet_lower for kw in promo_keywords):
            return "promo"

        # Social indicators
        social_keywords = [
            "liked your",
            "commented on",
            "mentioned you",
            "friend request",
            "connection",
        ]
        social_domains = ["facebook.com", "twitter.com", "linkedin.com", "instagram.com"]
        if any(kw in subject_lower or kw in snippet_lower for kw in social_keywords):
            return "social"
        if any(domain in from_lower for domain in social_domains):
            return "social"

        # Newsletter indicators
        if "unsubscribe" in snippet_lower or "newsletter" in subject_lower:
            return "newsletter"

        # Important indicators
        important_keywords = ["urgent", "important", "asap", "action required", "deadline"]
        if any(kw in subject_lower for kw in important_keywords):
            return "important"
        if getattr(email, "importance_score", 0.5) >= 0.8:
            return "important"

        return "general"

    def summarize(self, email_id: str, *, user_id: str | None = None) -> str:
        """
        Get a simple summary of an email (stub implementation).
        Currently returns a formatted snippet from the requesting user's inbox.
        """
        validated = require_user_id(user_id)

        if not self.connected:
            return "Email service not connected"

        for email in self._user_inbox(validated):
            if email.id == email_id:
                return f"From: {email.from_addr}\nSubject: {email.subject}\n\n{email.snippet}"

        return f"Email not found: {email_id}"

    def _triage_summary(self, email: EmailSummary) -> str:
        return f"{email.subject} (from {email.from_addr})"

    def _summary_dict(self, email: EmailSummary) -> dict[str, Any]:
        return {
            "id": email.id,
            "from_addr": email.from_addr,
            "subject": email.subject,
            "received_at": (
                email.received_at.isoformat()
                if isinstance(email.received_at, datetime)
                else str(email.received_at)
            ),
            "labels": list(email.labels or []),
            "importance_score": float(getattr(email, "importance_score", 0.5)),
            "category": getattr(email, "category", None),
        }

    def get_all_emails(self) -> list[EmailSummary]:
        """Get all emails from the loaded fixture (stub/testing)."""
        return self._mock_emails.copy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _envelope_to_summary(env: object) -> EmailSummary:
        """Convert an ``EmailEnvelope`` from a backend to ``EmailSummary``."""
        return EmailSummary(
            id=getattr(env, "message_id", ""),
            from_addr=getattr(env, "from_addr", ""),
            subject=getattr(env, "subject", ""),
            snippet=getattr(env, "snippet", ""),
            received_at=getattr(env, "received_at", datetime.now(UTC)),
            labels=list(getattr(env, "labels", [])),
            importance_score=0.5,
        )


# Global email service instance
_email_service: EmailService | None = None


def _create_configured_email_service() -> EmailService:
    """Create an EmailService wired to the per-user account resolver.

    Backends are resolved lazily per validated user and authorized account;
    no backend or credential is touched at construction time.

    Raises:
        IntegrationNotConfiguredError: when no email accounts are configured.
    """
    resolver = EmailAccountResolver.load()
    if not resolver.has_configured_accounts():
        raise IntegrationNotConfiguredError("Email: not configured")
    # No resolver is injected: the service loads it lazily and refreshes it
    # when the config file changes, so account revocations take effect in
    # long-lived processes without a restart.
    return EmailService()


def get_email_service() -> EmailService:
    """Get the global email service instance.

    The returned service enforces per-user account ownership internally;
    every operation requires a validated ``user_id``.

    Raises:
        IntegrationNotConfiguredError: when no email accounts are configured.
    """
    global _email_service
    if _email_service is None:
        _email_service = _create_configured_email_service()
    return _email_service


def set_email_service(service: EmailService) -> None:
    """Set the global email service instance (for testing)."""
    global _email_service
    _email_service = service


__all__ = [
    "EmailSummary",
    "EmailMessage",
    "EmailService",
    "get_email_service",
    "set_email_service",
]
