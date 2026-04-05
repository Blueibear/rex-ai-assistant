"""Rex outbound calling via Twilio REST API.

Public API:
    make_call(to_number, message=None, confirmed=False) -> dict
    lookup_contact(name, contacts_file) -> str | None
    log_call(to_number, outcome) -> None
    detect_call_intent(text) -> tuple[str | None, str | None]

Gracefully disabled (returns error dict) when Twilio credentials are absent.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

_OUTBOUND_LOG_FILE = "data/outbound_calls.log"

# Env-var names (loaded from .env by python-dotenv on startup).
_ENV_ACCOUNT_SID = "TWILIO_ACCOUNT_SID"
_ENV_AUTH_TOKEN = "TWILIO_AUTH_TOKEN"
_ENV_PHONE_NUMBER = "TWILIO_PHONE_NUMBER"

# Patterns for detecting outbound call intent.
_CALL_INTENT_PATTERNS = [
    re.compile(
        r"(?:please\s+)?call\s+(?:my\s+)?(.+?)(?:\s+for\s+me)?$",
        re.IGNORECASE,
    ),
    re.compile(r"dial\s+(.+)$", re.IGNORECASE),
    re.compile(r"phone\s+(.+)$", re.IGNORECASE),
    re.compile(r"ring\s+(.+)$", re.IGNORECASE),
]

# E.164-ish phone number pattern (digits, spaces, dashes, parens, + prefix).
_PHONE_RE = re.compile(r"^\+?[\d\s\-().]{7,20}$")


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------


def _get_credentials() -> tuple[str, str, str] | None:
    """Return (account_sid, auth_token, phone_number) or None if absent."""
    sid = os.environ.get(_ENV_ACCOUNT_SID, "").strip()
    token = os.environ.get(_ENV_AUTH_TOKEN, "").strip()
    phone = os.environ.get(_ENV_PHONE_NUMBER, "").strip()
    if not (sid and token and phone):
        return None
    return sid, token, phone


def is_configured() -> bool:
    """Return True if Twilio credentials are present in the environment."""
    return _get_credentials() is not None


# ---------------------------------------------------------------------------
# Contact lookup
# ---------------------------------------------------------------------------


def lookup_contact(name: str, contacts_file: str) -> str | None:
    """Return a phone number for *name* from *contacts_file*, or None.

    Supports two file formats:
    - JSON: ``{"contacts": [{"name": "...", "phone": "..."}, ...]}``
    - vCard (.vcf): ``BEGIN:VCARD`` blocks with ``FN:`` and ``TEL:`` lines.
    """
    if not contacts_file or not os.path.isfile(contacts_file):
        return None

    name_lower = name.strip().lower()

    try:
        if contacts_file.lower().endswith(".vcf"):
            return _lookup_vcf(name_lower, contacts_file)
        return _lookup_json(name_lower, contacts_file)
    except Exception as exc:
        logger.warning("Failed to read contacts file %s: %s", contacts_file, exc)
        return None


def _lookup_json(name_lower: str, contacts_file: str) -> str | None:
    with open(contacts_file, encoding="utf-8") as fh:
        data: Any = json.load(fh)

    contacts = data.get("contacts", [])
    for entry in contacts:
        entry_name = str(entry.get("name", "")).lower()
        if name_lower in entry_name or entry_name in name_lower:
            phone = str(entry.get("phone", "")).strip()
            if phone:
                return phone
    return None


def _lookup_vcf(name_lower: str, contacts_file: str) -> str | None:
    with open(contacts_file, encoding="utf-8") as fh:
        text = fh.read()

    current_fn: str | None = None
    current_tel: str | None = None

    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("FN:"):
            current_fn = line[3:].strip().lower()
            current_tel = None
        elif line.upper().startswith("TEL") and ":" in line:
            current_tel = line.split(":", 1)[1].strip()
        elif line.upper() == "END:VCARD":
            if current_fn and current_tel:
                if name_lower in current_fn or current_fn in name_lower:
                    return current_tel
            current_fn = None
            current_tel = None

    return None


# ---------------------------------------------------------------------------
# Call logging
# ---------------------------------------------------------------------------


def log_call(to_number: str, outcome: str, message: str | None = None) -> None:
    """Append an outbound call record to the log file."""
    os.makedirs(os.path.dirname(_OUTBOUND_LOG_FILE), exist_ok=True)
    timestamp = datetime.datetime.now().isoformat()
    record: dict[str, Any] = {
        "timestamp": timestamp,
        "to": to_number,
        "outcome": outcome,
    }
    if message:
        record["message"] = message

    try:
        with open(_OUTBOUND_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.info("Outbound call logged: to=%s outcome=%s", to_number, outcome)
    except Exception as exc:
        logger.warning("Failed to write outbound call log: %s", exc)


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------


def detect_call_intent(text: str) -> tuple[str | None, str | None]:
    """Parse a user utterance for outbound call intent.

    Returns ``(target, message_hint)`` where *target* is a contact name or
    phone number extracted from *text*, and *message_hint* is an optional
    message to play.  Returns ``(None, None)`` if no call intent detected.
    """
    text = text.strip()
    for pattern in _CALL_INTENT_PATTERNS:
        match = pattern.search(text)
        if match:
            target = match.group(1).strip()
            return target, None
    return None, None


def is_phone_number(target: str) -> bool:
    """Return True if *target* looks like a phone number rather than a name."""
    return bool(_PHONE_RE.match(target.strip()))


# ---------------------------------------------------------------------------
# make_call
# ---------------------------------------------------------------------------


def make_call(
    to_number: str,
    message: str | None = None,
    *,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Initiate an outbound call via Twilio REST API.

    Args:
        to_number: E.164 phone number to dial (e.g. ``"+15551234567"``).
        message: Optional TTS message to play when the call connects.
                 If None, Rex's inbound conversation loop URL is used instead.
        confirmed: Must be True; if False, returns a confirmation-request
                   dict without dialing so the caller can prompt the user.

    Returns:
        A result dict with keys ``ok`` (bool) and either ``call_sid`` (success)
        or ``error`` (failure) and ``needs_confirmation`` (pre-dial).
    """
    if not confirmed:
        logger.debug("make_call: confirmation required before dialing %s", to_number)
        return {
            "ok": False,
            "needs_confirmation": True,
            "to": to_number,
            "message": message,
            "prompt": (
                f"Are you sure you want Rex to call {to_number}? " "Reply 'yes' to confirm."
            ),
        }

    creds = _get_credentials()
    if not creds:
        logger.warning("make_call: Twilio not configured — cannot place call")
        log_call(to_number, "error:not_configured", message)
        return {"ok": False, "error": "Twilio integration not configured."}

    account_sid, auth_token, from_number = creds

    # Build TwiML for the call.
    if message:
        twiml = _build_say_twiml(message)
        twiml_param: dict[str, Any] = {"twiml": twiml}
    else:
        # Point the call back at Rex's inbound gather loop for conversation.
        twiml_param = {"url": _get_rex_base_url() + "/telephony/inbound/call"}

    try:
        from twilio.rest import Client

        client = Client(account_sid, auth_token)
        call = client.calls.create(
            to=to_number,
            from_=from_number,
            **twiml_param,
        )
        logger.info("Outbound call initiated: sid=%s to=%s", call.sid, to_number)
        log_call(to_number, f"initiated:{call.sid}", message)
        return {"ok": True, "call_sid": call.sid, "to": to_number}

    except ImportError:
        logger.warning("twilio package not installed; cannot place call")
        log_call(to_number, "error:twilio_not_installed", message)
        return {"ok": False, "error": "twilio package is not installed."}

    except Exception as exc:
        logger.exception("make_call failed for %s: %s", to_number, exc)
        log_call(to_number, f"error:{exc}", message)
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_say_twiml(text: str) -> str:
    """Return TwiML that speaks *text* and hangs up."""
    try:
        from twilio.twiml.voice_response import VoiceResponse

        resp = VoiceResponse()
        resp.say(text)
        resp.hangup()
        return str(resp)
    except ImportError:
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return (
            "<?xml version='1.0' encoding='UTF-8'?>"
            f"<Response><Say>{safe}</Say><Hangup/></Response>"
        )


def _get_rex_base_url() -> str:
    """Return Rex's base URL for webhook callbacks.

    Reads ``REX_BASE_URL`` from the environment; falls back to localhost.
    """
    return os.environ.get("REX_BASE_URL", "http://localhost:5000").rstrip("/")


__all__ = [
    "make_call",
    "lookup_contact",
    "log_call",
    "detect_call_intent",
    "is_phone_number",
    "is_configured",
]
