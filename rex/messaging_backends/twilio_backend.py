"""Twilio SMS backend using the ``requests`` library.

This backend calls the Twilio REST API directly via ``requests`` so that the
heavy ``twilio`` package remains an optional install.  If the ``twilio``
package *is* installed it is **not** used — ``requests`` is sufficient for
the small API surface we need (send SMS, list recent messages).

Credentials required (via CredentialManager):
- ``twilio_account_sid``  – Twilio Account SID
- ``twilio_auth_token``   – Twilio Auth Token

Non-secret config (in ``config/rex_config.json``):
- ``messaging.accounts[].from_number``  – The Twilio phone number to send from
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

from rex.credentials import CredentialManager, mask_token
from rex.messaging_backends.base import InboundSms, SmsBackend, SmsSendResult

logger = logging.getLogger(__name__)

_TWILIO_API_BASE = "https://api.twilio.com/2010-04-01"


class TwilioSmsBackend(SmsBackend):
    """Real SMS backend using the Twilio REST API via ``requests``.

    Args:
        account_sid: Twilio Account SID.
        auth_token: Twilio Auth Token.
        default_from: Default sender phone number (E.164).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        *,
        account_sid: str,
        auth_token: str,
        default_from: str,
        timeout: float = 15.0,
    ) -> None:
        if not account_sid or not auth_token:
            raise ValueError(
                "Twilio credentials are required (account_sid and auth_token). "
                "Set them via CredentialManager or environment variables."
            )
        self._account_sid = account_sid
        self._auth_token = auth_token
        self._default_from = default_from
        self._timeout = timeout
        self._base_url = f"{_TWILIO_API_BASE}/Accounts/{account_sid}"
        logger.info(
            "TwilioSmsBackend initialized (account=%s, from=%s)",
            mask_token(account_sid),
            default_from,
        )

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
        """Send an SMS via the Twilio Messages API."""
        from_num = from_number or self._default_from
        url = f"{self._base_url}/Messages.json"
        payload = {
            "To": to,
            "From": from_num,
            "Body": body,
        }
        try:
            resp = requests.post(
                url,
                data=payload,
                auth=(self._account_sid, self._auth_token),
                timeout=self._timeout,
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                sid = data.get("sid", "")
                logger.info("Twilio SMS sent successfully (sid=%s)", sid)
                return SmsSendResult(ok=True, message_sid=sid)

            # Twilio error response
            error_msg = _extract_twilio_error(resp)
            logger.warning(
                "Twilio SMS send failed (status=%d): %s",
                resp.status_code,
                error_msg,
            )
            return SmsSendResult(ok=False, error=error_msg)

        except requests.Timeout:
            logger.warning("Twilio SMS send timed out")
            return SmsSendResult(ok=False, error="Request timed out")
        except requests.ConnectionError as exc:
            logger.warning("Twilio SMS send connection error: %s", exc)
            return SmsSendResult(ok=False, error=f"Connection error: {exc}")
        except Exception as exc:
            logger.error("Twilio SMS send unexpected error: %s", exc)
            return SmsSendResult(ok=False, error=str(exc))

    def fetch_recent_inbound(self, *, limit: int = 20) -> list[InboundSms]:
        """Fetch recent inbound messages via the Twilio Messages API."""
        url = f"{self._base_url}/Messages.json"
        params = {
            "To": self._default_from,
            "PageSize": min(limit, 100),
        }
        try:
            resp = requests.get(
                url,
                params=params,  # type: ignore[arg-type]
                auth=(self._account_sid, self._auth_token),
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                error_msg = _extract_twilio_error(resp)
                logger.warning(
                    "Twilio fetch inbound failed (status=%d): %s",
                    resp.status_code,
                    error_msg,
                )
                return []

            data = resp.json()
            messages_raw = data.get("messages", [])
            results: list[InboundSms] = []
            for msg in messages_raw[:limit]:
                ts = _parse_twilio_dt(msg.get("date_created"))
                results.append(
                    InboundSms(
                        sid=msg.get("sid", ""),
                        from_number=msg.get("from", ""),
                        to_number=msg.get("to", ""),
                        body=msg.get("body", ""),
                        received_at=ts,
                    )
                )
            return results

        except requests.Timeout:
            logger.warning("Twilio fetch inbound timed out")
            return []
        except requests.ConnectionError as exc:
            logger.warning("Twilio fetch inbound connection error: %s", exc)
            return []
        except Exception as exc:
            logger.error("Twilio fetch inbound unexpected error: %s", exc)
            return []


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _extract_twilio_error(resp: requests.Response) -> str:
    """Extract an error message from a Twilio error response."""
    try:
        data = resp.json()
        code = data.get("code", "")
        message = data.get("message", "")
        return f"Twilio error {code}: {message}" if code else message or resp.text[:200]
    except Exception:
        return resp.text[:200] if resp.text else f"HTTP {resp.status_code}"


def _parse_twilio_dt(value: str | None) -> datetime:
    """Parse a Twilio-style datetime string."""
    if not value:
        return datetime.now(timezone.utc)
    try:
        # Twilio uses RFC 2822 format; Python's email.utils can parse it
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(value)
    except Exception:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return datetime.now(timezone.utc)


def create_twilio_backend_from_credentials(
    credential_manager: CredentialManager,
    credential_ref: str,
    from_number: str,
    timeout: float = 15.0,
) -> TwilioSmsBackend:
    """Create a ``TwilioSmsBackend`` from a CredentialManager reference.

    The credential token is expected to be in ``account_sid:auth_token`` format.

    Args:
        credential_manager: The CredentialManager instance.
        credential_ref: Key to look up in CredentialManager.
        from_number: Default sender phone number.
        timeout: HTTP request timeout in seconds.

    Returns:
        A configured ``TwilioSmsBackend``.

    Raises:
        ValueError: If credentials are missing or malformed.
    """
    token = credential_manager.get_token(credential_ref)
    if not token:
        raise ValueError(
            f"No Twilio credentials found for '{credential_ref}'. "
            "Set the credential via environment variable or config/credentials.json "
            "in 'account_sid:auth_token' format."
        )
    if ":" not in token:
        raise ValueError(
            f"Twilio credential for '{credential_ref}' must be in "
            "'account_sid:auth_token' format."
        )
    account_sid, auth_token = token.split(":", 1)
    return TwilioSmsBackend(
        account_sid=account_sid,
        auth_token=auth_token,
        default_from=from_number,
        timeout=timeout,
    )


__all__ = [
    "TwilioSmsBackend",
    "create_twilio_backend_from_credentials",
]
