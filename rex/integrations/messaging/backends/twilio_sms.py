"""Twilio SMS send adapter implementing the transport-layer SMSBackend (US-211).

Uses the optional ``twilio`` package.  An informative ``ImportError`` is raised
at instantiation time if the package is not installed.

Credentials are resolved at send-time via :class:`~rex.credentials.CredentialManager`
using three keys:

- ``twilio_account_sid``
- ``twilio_auth_token``
- ``twilio_from_number``

Explicit constructor arguments override credential lookups.

**No secret is ever logged**, at any log level.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol, cast

from rex.credentials import CredentialManager
from rex.integrations.messaging.backends.base import SMSBackend

logger = logging.getLogger(__name__)


class _TwilioMessagesProtocol(Protocol):
    def create(self, *, to: str, from_: str, body: str) -> object:
        ...


class _TwilioClientProtocol(Protocol):
    @property
    def messages(self) -> _TwilioMessagesProtocol:
        ...


class SMSSendError(Exception):
    """Raised when an SMS send fails (4xx Twilio error, timeout, or other).

    Provides a descriptive message that **never includes credentials**.
    """


class TwilioSMSBackend(SMSBackend):
    """SMS send backend using the Twilio REST API.

    The ``twilio`` Python package is an optional dependency.  If it is not
    installed, ``ImportError`` is raised at construction time with a
    human-readable hint.

    Args:
        account_sid:    Twilio account SID.  When *None* (default) the value
                        is resolved via ``CredentialManager.get_token("twilio_account_sid")``.
        auth_token:     Twilio auth token.  When *None* (default) resolved
                        from ``"twilio_auth_token"``.
        from_number:    Sender phone number in E.164 format.  When *None*
                        resolved from ``"twilio_from_number"``.
        twilio_client_factory:
                        Optional callable that accepts ``(account_sid, auth_token)``
                        and returns a Twilio ``Client``-compatible object.
                        Inject for unit tests to avoid live network calls.
    """

    def __init__(
        self,
        *,
        account_sid: str | None = None,
        auth_token: str | None = None,
        from_number: str | None = None,
        twilio_client_factory: Callable[[str, str], _TwilioClientProtocol] | None = None,
    ) -> None:
        # Fail fast if twilio is not installed (and no factory injected)
        if twilio_client_factory is None:
            try:
                import twilio  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "The 'twilio' package is required for TwilioSMSBackend. "
                    "Install it with: pip install twilio"
                ) from exc

        self._account_sid = account_sid
        self._auth_token = auth_token
        self._from_number = from_number
        self._client_factory = twilio_client_factory

    # ------------------------------------------------------------------
    # Credential resolution
    # ------------------------------------------------------------------

    def _resolve_credentials(self) -> tuple[str, str, str]:
        """Return (account_sid, auth_token, from_number) — never logged.

        Explicit constructor args take precedence; falls back to
        CredentialManager for any missing values.

        Raises:
            SMSSendError: If any required credential is missing.
        """
        mgr = CredentialManager()

        account_sid = self._account_sid or mgr.get_token("twilio_account_sid") or ""
        auth_token = self._auth_token or mgr.get_token("twilio_auth_token") or ""
        from_number = self._from_number or mgr.get_token("twilio_from_number") or ""

        missing = [
            name
            for name, val in (
                ("twilio_account_sid", account_sid),
                ("twilio_auth_token", auth_token),
                ("twilio_from_number", from_number),
            )
            if not val
        ]
        if missing:
            raise SMSSendError(
                f"Missing Twilio credentials: {', '.join(missing)}. "
                "Set them via CredentialManager or pass them to TwilioSMSBackend."
            )

        return account_sid, auth_token, from_number

    # ------------------------------------------------------------------
    # SMSBackend interface
    # ------------------------------------------------------------------

    def send(self, to: str, body: str) -> None:
        """Send an SMS via Twilio.

        Args:
            to:   Recipient phone number in E.164 format.
            body: Message text.

        Raises:
            SMSSendError: On Twilio 4xx error, network timeout, or missing
                          credentials.
        """
        account_sid, auth_token, from_number = self._resolve_credentials()

        try:
            client = self._build_client(account_sid, auth_token)
            client.messages.create(to=to, from_=from_number, body=body)
            logger.info("SMS sent via Twilio to %s (%d chars)", to, len(body))
        except SMSSendError:
            raise
        except Exception as exc:
            self._handle_twilio_exc(exc, to)

    def receive(self) -> list[dict]:
        """Not implemented — TwilioSMSBackend is send-only.

        Inbound Twilio webhooks are handled at the HTTP layer; polling is
        not supported by this backend.
        """
        logger.debug("TwilioSMSBackend.receive() called — returning empty list (send-only)")
        return []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_client(self, account_sid: str, auth_token: str) -> _TwilioClientProtocol:
        if self._client_factory is not None:
            return self._client_factory(account_sid, auth_token)
        from twilio.rest import Client

        return cast(_TwilioClientProtocol, Client(account_sid, auth_token))

    @staticmethod
    def _handle_twilio_exc(exc: Exception, to: str) -> None:
        """Translate Twilio / network exceptions into SMSSendError.

        The original exception message is included only when it does not
        contain the auth token (best-effort secret scrub).
        """
        exc_type = type(exc).__name__

        # Twilio SDK raises twilio.base.exceptions.TwilioRestException
        # for API errors; detect by duck-typing.
        twilio_code = getattr(exc, "code", None)
        twilio_status = getattr(exc, "status", None)

        if twilio_status is not None and twilio_code is not None:
            raise SMSSendError(
                f"Twilio API error {twilio_status} (code {twilio_code}) " f"sending to {to}"
            ) from exc

        # TimeoutError / socket timeout
        if isinstance(exc, (TimeoutError, OSError)) or "timeout" in str(exc).lower():
            raise SMSSendError(f"Twilio request timed out sending to {to}") from exc

        raise SMSSendError(f"SMS send failed ({exc_type}) sending to {to}: {exc}") from exc


__all__ = ["SMSSendError", "TwilioSMSBackend"]
