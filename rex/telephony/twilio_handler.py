"""Twilio inbound webhook handler for Rex.

Provides Flask Blueprint routes:
    POST /telephony/inbound/call      — answers with TwiML greeting + <Gather>
    POST /telephony/inbound/gather    — conversation loop (STT → LLM → TTS)
    POST /telephony/inbound/voicemail — saves transcribed voicemail to disk
    POST /telephony/inbound/sms       — passes SMS body to LLM, replies via Twilio

Webhook signature validation uses twilio.request_validator.RequestValidator
to prevent spoofed requests.

Gracefully disabled (returns 503) when Twilio credentials are absent.
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flask import Blueprint

logger = logging.getLogger(__name__)

_INBOUND_CALL_GREETING = "Hi, you've reached Rex. How can I help?"
_MAX_TURNS = 5
_VOICEMAIL_DIR = "data/voicemail"

_VOICEMAIL_KEYWORDS: frozenset[str] = frozenset(
    ["leave a message", "leave message", "voicemail", "take a message"]
)
_TRANSFER_KEYWORDS: frozenset[str] = frozenset(
    ["transfer", "speak to a human", "speak to someone", "real person", "talk to a person"]
)

# Env-var names (loaded from .env by python-dotenv on startup).
_ENV_ACCOUNT_SID = "TWILIO_ACCOUNT_SID"
_ENV_AUTH_TOKEN = "TWILIO_AUTH_TOKEN"
_ENV_PHONE_NUMBER = "TWILIO_PHONE_NUMBER"
_ENV_TRANSFER_NUMBER = "TWILIO_TRANSFER_NUMBER"


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


def _get_transfer_number() -> str | None:
    """Return the configured call-transfer number from the environment, or None."""
    return os.environ.get(_ENV_TRANSFER_NUMBER, "").strip() or None


# ---------------------------------------------------------------------------
# Signature validation (module-level for testability)
# ---------------------------------------------------------------------------


def _validate_twilio_signature(auth_token: str) -> bool:
    """Return True if the current Flask request has a valid Twilio signature.

    Skips validation (returns True) when the twilio package is not installed.
    """
    try:
        from flask import request
        from twilio.request_validator import RequestValidator

        validator = RequestValidator(auth_token)
        url = request.url
        post_data = request.form.to_dict()
        signature = request.headers.get("X-Twilio-Signature", "")
        return validator.validate(url, post_data, signature)
    except ImportError:
        logger.warning("twilio package not installed; skipping signature validation")
        return True
    except Exception as exc:
        logger.warning("Signature validation error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Conversation intent helpers
# ---------------------------------------------------------------------------


def _detect_voicemail_intent(text: str) -> bool:
    """Return True if *text* signals the caller wants to leave a voicemail."""
    lowered = text.lower()
    return any(kw in lowered for kw in _VOICEMAIL_KEYWORDS)


def _detect_transfer_intent(text: str) -> bool:
    """Return True if *text* signals the caller wants to be transferred."""
    lowered = text.lower()
    return any(kw in lowered for kw in _TRANSFER_KEYWORDS)


# ---------------------------------------------------------------------------
# Voicemail
# ---------------------------------------------------------------------------


def _save_voicemail(caller_number: str, text: str) -> str:
    """Save *text* as a voicemail transcript and return the file path."""
    os.makedirs(_VOICEMAIL_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_caller = caller_number.lstrip("+").replace(" ", "_")
    path = os.path.join(_VOICEMAIL_DIR, f"{safe_caller}_{timestamp}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"From: {caller_number}\n")
        fh.write(f"Time: {timestamp}\n\n")
        fh.write(text)
    logger.info("Voicemail saved to %s", path)
    return path


# ---------------------------------------------------------------------------
# TwiML builders
# ---------------------------------------------------------------------------


def _build_gather_twiml(say_text: str, gather_action: str) -> str:
    """Return TwiML <Gather input=speech> wrapping a <Say> prompt.

    Falls back to hand-crafted XML if the twilio package is not installed.
    """
    try:
        from twilio.twiml.voice_response import Gather, VoiceResponse

        resp = VoiceResponse()
        gather = Gather(
            input="speech",
            action=gather_action,
            speech_timeout="auto",
            method="POST",
        )
        gather.say(say_text)
        resp.append(gather)
        resp.say("I didn't catch that. Goodbye.")
        resp.hangup()
        return str(resp)
    except ImportError:
        safe_text = (
            say_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        return (
            "<?xml version='1.0' encoding='UTF-8'?>"
            "<Response>"
            f'<Gather input="speech" action="{gather_action}"'
            ' speechTimeout="auto" method="POST">'
            f"<Say>{safe_text}</Say>"
            "</Gather>"
            "<Say>I didn't catch that. Goodbye.</Say>"
            "<Hangup/>"
            "</Response>"
        )


def _build_hangup_twiml(say_text: str) -> str:
    """Return TwiML that says *say_text* then hangs up."""
    try:
        from twilio.twiml.voice_response import VoiceResponse

        resp = VoiceResponse()
        resp.say(say_text)
        resp.hangup()
        return str(resp)
    except ImportError:
        safe_text = (
            say_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        return (
            "<?xml version='1.0' encoding='UTF-8'?>"
            f"<Response><Say>{safe_text}</Say><Hangup/></Response>"
        )


# ---------------------------------------------------------------------------
# Blueprint factory
# ---------------------------------------------------------------------------


def create_blueprint() -> Blueprint:
    """Create and return the Twilio webhook Flask Blueprint."""
    from flask import Blueprint, Response, request

    bp = Blueprint("telephony", __name__, url_prefix="/telephony")

    def _not_configured() -> tuple[Any, int]:
        from flask import jsonify

        return (
            jsonify({"error": "Twilio integration not configured."}),
            503,
        )

    def _require_signature(auth_token: str) -> Response | None:
        """Return a 403 Response if signature invalid, else None."""
        if not _validate_twilio_signature(auth_token):
            return Response("Forbidden", status=403)
        return None

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @bp.route("/inbound/call", methods=["POST"])
    def inbound_call() -> Any:
        """Answer an inbound Twilio call with a TwiML greeting + <Gather>."""
        creds = _get_credentials()
        if not creds:
            return _not_configured()

        _, auth_token, _ = creds
        sig_error = _require_signature(auth_token)
        if sig_error:
            return sig_error

        twiml = _build_gather_twiml(
            _INBOUND_CALL_GREETING,
            "/telephony/inbound/gather?turn=1",
        )
        return Response(twiml, mimetype="application/xml")

    @bp.route("/inbound/gather", methods=["POST"])
    def inbound_gather() -> Any:
        """Handle <Gather> callback: STT → LLM → TTS conversation loop."""
        creds = _get_credentials()
        if not creds:
            return _not_configured()

        _, auth_token, _ = creds
        sig_error = _require_signature(auth_token)
        if sig_error:
            return sig_error

        speech_result = request.form.get("SpeechResult", "").strip()
        caller = request.form.get("From", "unknown")

        try:
            turn = int(request.args.get("turn", "1"))
        except ValueError:
            turn = 1

        # Graceful end after max turns
        if turn > _MAX_TURNS:
            twiml = _build_hangup_twiml("Thanks for calling Rex. Goodbye!")
            return Response(twiml, mimetype="application/xml")

        # No speech detected — prompt again (same turn)
        if not speech_result:
            twiml = _build_gather_twiml(
                "I didn't hear anything. Could you repeat that?",
                f"/telephony/inbound/gather?turn={turn}",
            )
            return Response(twiml, mimetype="application/xml")

        # Voicemail intent
        if _detect_voicemail_intent(speech_result):
            try:
                from twilio.twiml.voice_response import VoiceResponse

                resp = VoiceResponse()
                resp.say("Sure, please leave your message after the tone.")
                resp.record(
                    action="/telephony/inbound/voicemail",
                    method="POST",
                    max_length=120,
                    transcribe=True,
                    transcribe_callback="/telephony/inbound/voicemail",
                )
                twiml = str(resp)
            except ImportError:
                twiml = (
                    "<?xml version='1.0' encoding='UTF-8'?>"
                    "<Response>"
                    "<Say>Sure, please leave your message after the tone.</Say>"
                    '<Record action="/telephony/inbound/voicemail" method="POST"'
                    ' maxLength="120" transcribe="true"'
                    ' transcribeCallback="/telephony/inbound/voicemail"/>'
                    "</Response>"
                )
            return Response(twiml, mimetype="application/xml")

        # Transfer intent
        transfer_number = _get_transfer_number()
        if _detect_transfer_intent(speech_result) and transfer_number:
            try:
                from twilio.twiml.voice_response import Dial, VoiceResponse

                resp = VoiceResponse()
                resp.say("Connecting you now.")
                dial = Dial()
                dial.number(transfer_number)
                resp.append(dial)
                twiml = str(resp)
            except ImportError:
                twiml = (
                    "<?xml version='1.0' encoding='UTF-8'?>"
                    "<Response>"
                    "<Say>Connecting you now.</Say>"
                    f"<Dial>{transfer_number}</Dial>"
                    "</Response>"
                )
            return Response(twiml, mimetype="application/xml")

        # Normal conversation: call LLM, return next <Gather>
        reply = _generate_reply(speech_result)
        next_turn = turn + 1
        twiml = _build_gather_twiml(reply, f"/telephony/inbound/gather?turn={next_turn}")
        return Response(twiml, mimetype="application/xml")

    @bp.route("/inbound/voicemail", methods=["POST"])
    def inbound_voicemail() -> Any:
        """Receive Twilio voicemail transcription callback and save to disk."""
        creds = _get_credentials()
        if not creds:
            return _not_configured()

        # Accept either TranscriptionText (async callback) or SpeechResult
        transcription = (
            request.form.get("TranscriptionText")
            or request.form.get("SpeechResult")
            or "(No transcription available)"
        )
        caller = request.form.get("From", "unknown")
        _save_voicemail(caller, transcription)

        twiml = _build_hangup_twiml("Your message has been saved. Goodbye!")
        return Response(twiml, mimetype="application/xml")

    @bp.route("/inbound/sms", methods=["POST"])
    def inbound_sms() -> Any:
        """Receive an inbound SMS and reply with an LLM-generated response."""
        creds = _get_credentials()
        if not creds:
            return _not_configured()

        account_sid, auth_token, from_number = creds
        sig_error = _require_signature(auth_token)
        if sig_error:
            return sig_error

        body = request.form.get("Body", "").strip()
        from_caller = request.form.get("From", "")

        if not body:
            logger.info("Received empty SMS from %s; ignoring.", from_caller)
            return Response("", status=204)

        reply = _generate_sms_reply(body)

        # Send reply via Twilio REST API.
        _send_sms(account_sid, auth_token, from_number=from_number, to=from_caller, body=reply)

        # Return empty TwiML acknowledgement — we reply out-of-band above.
        try:
            from twilio.twiml.messaging_response import MessagingResponse

            twiml = MessagingResponse()
            return Response(str(twiml), mimetype="application/xml")
        except ImportError:
            twiml_str = "<?xml version='1.0' encoding='UTF-8'?><Response/>"
            return Response(twiml_str, mimetype="application/xml")

    return bp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_reply(text: str) -> str:
    """Pass *text* through the Rex LLM and return the reply string."""
    try:
        import asyncio

        from rex.assistant import Assistant
        from rex.config import load_config

        cfg = load_config()
        assistant = Assistant(config=cfg)
        loop = asyncio.new_event_loop()
        try:
            reply: str = loop.run_until_complete(assistant.generate_reply(text))
        finally:
            loop.close()
        return reply
    except Exception as exc:
        logger.exception("Failed to generate reply: %s", exc)
        return f"I'm having trouble right now. (Error: {exc})"


def _generate_sms_reply(text: str) -> str:
    """Pass *text* through the Rex LLM and return the reply string."""
    return _generate_reply(text)


def _send_sms(account_sid: str, auth_token: str, *, from_number: str, to: str, body: str) -> None:
    """Send an SMS via the Twilio REST API."""
    try:
        from twilio.rest import Client

        client = Client(account_sid, auth_token)
        client.messages.create(body=body, from_=from_number, to=to)
        logger.info("SMS reply sent to %s", to)
    except ImportError:
        logger.warning("twilio package not installed; cannot send SMS reply")
    except Exception as exc:
        logger.exception("Failed to send SMS to %s: %s", to, exc)


__all__ = [
    "create_blueprint",
    "is_configured",
    "_validate_twilio_signature",
    "_detect_voicemail_intent",
    "_detect_transfer_intent",
    "_save_voicemail",
    "_get_transfer_number",
    "_build_gather_twiml",
    "_build_hangup_twiml",
]
