"""Integrations, calendar, email, SMS, capabilities, and tools routes."""

from __future__ import annotations

import os
from typing import Any

from flask import Blueprint

_CALENDAR_NO_USER_ERROR = (
    "No active user for calendar. Set one with 'rex identify --user <id>' "
    "or pass an explicit ?user= parameter."
)

_OUTLOOK_CALENDAR_UNSUPPORTED = (
    "Outlook calendar sync is not implemented yet. The current Outlook "
    "settings only store app credentials; Rex cannot read or write "
    "Outlook events until Microsoft Graph OAuth token support is added."
)


def _calendar_events_response() -> Any:
    """Build the /api/calendar/events response for the resolved user only.

    Fails closed (403) without a valid identity; provider and credential
    selection are per user via ``rex.calendar_accounts`` (issue #303).
    """
    from datetime import UTC, datetime, timedelta

    from flask import jsonify, request

    from rex.identity import resolve_active_user
    from rex.integrations.calendar_service import create_calendar_service_for_user

    try:
        from rex.config_manager import load_config as _load_raw_config

        raw_config = _load_raw_config()
    except Exception:
        raw_config = {}

    explicit = str(request.args.get("user") or "").strip() or None
    try:
        user_id = resolve_active_user(explicit, config=raw_config)
    except ValueError:
        user_id = None
    if not user_id:
        return (
            jsonify(
                {"ok": False, "events": [], "configured": False, "error": _CALENDAR_NO_USER_ERROR}
            ),
            403,
        )

    try:
        svc, provider = create_calendar_service_for_user(user_id, raw_config)
    except PermissionError:
        return (
            jsonify(
                {"ok": False, "events": [], "configured": False, "error": _CALENDAR_NO_USER_ERROR}
            ),
            403,
        )

    if provider == "outlook":
        return (
            jsonify(
                {
                    "ok": False,
                    "events": [],
                    "configured": True,
                    "error": _OUTLOOK_CALENDAR_UNSUPPORTED,
                }
            ),
            501,
        )

    if svc is None:
        return jsonify({"ok": True, "events": [], "configured": False}), 200

    try:
        start_str = request.args.get("start", "")
        end_str = request.args.get("end", "")
        start = datetime.fromisoformat(start_str) if start_str else datetime.now(UTC)
        end = datetime.fromisoformat(end_str) if end_str else start + timedelta(days=30)
    except ValueError:
        start = datetime.now(UTC)
        end = start + timedelta(days=30)

    events = svc.get_events(start, end)

    def _event_to_dict(e: Any) -> dict:
        return {
            "id": e.id,
            "title": e.title,
            "start": e.start.isoformat(),
            "end": e.end.isoformat(),
            "location": e.location,
            "description": e.description,
            "attendees": list(e.attendees),
            "source": e.source,
            "is_all_day": e.is_all_day,
        }

    return (
        jsonify(
            {
                "ok": True,
                "events": [_event_to_dict(e) for e in events],
                "configured": True,
            }
        ),
        200,
    )


def create_blueprint() -> Blueprint:
    """Return the integrations and capabilities Blueprint."""
    bp = Blueprint("integrations", __name__)

    # ------------------------------------------------------------------
    # Integrations API (US-057)
    # ------------------------------------------------------------------

    @bp.route("/api/integrations", methods=["GET"])
    def _list_integrations() -> Any:
        """Return configured integrations with status and configure_url (public)."""
        from flask import jsonify

        from rex.config import load_config

        try:
            cfg = load_config()
        except Exception:
            return jsonify({"integrations": []}), 200

        search_configured = bool(
            os.getenv("SERPAPI_API_KEY") or os.getenv("BRAVE_API_KEY") or os.getenv("GOOGLE_CSE_ID")
        )

        integrations = [
            {
                "name": "Home Assistant",
                "key": "home_assistant",
                "configured": bool(cfg.ha_base_url and cfg.ha_token),
                "configure_url": "/settings/home-assistant",
            },
            {
                "name": "Email",
                "key": "email",
                "configured": cfg.email_provider not in ("none", ""),
                "configure_url": "/settings?section=integrations",
            },
            {
                "name": "Calendar",
                "key": "calendar",
                "configured": getattr(cfg, "calendar_provider", "none") not in ("none", ""),
                "configure_url": "/settings?section=integrations",
            },
            {
                "name": "SMS (Twilio)",
                "key": "sms",
                "configured": bool(
                    os.getenv("TWILIO_ACCOUNT_SID") or os.getenv("TWILIO_AUTH_TOKEN")
                ),
                "configure_url": "/settings?section=integrations",
            },
            {
                "name": "Telegram",
                "key": "telegram",
                "configured": bool(cfg.telegram_bot_token and cfg.telegram_chat_id),
                "configure_url": "/settings?section=integrations",
            },
            {
                "name": "Web Search",
                "key": "search",
                "configured": search_configured,
                "configure_url": "/settings?section=ai",
            },
            {
                "name": "MQTT",
                "key": "mqtt",
                "configured": bool(os.getenv("MQTT_BROKER_HOST")),
                "configure_url": "/settings?section=integrations",
            },
            {
                "name": "OpenAI",
                "key": "openai",
                "configured": bool(cfg.openai_api_key),
                "configure_url": "/settings?section=ai",
            },
            {
                "name": "Ollama",
                "key": "ollama",
                "configured": bool(cfg.ollama_base_url),
                "configure_url": "/settings?section=ai",
            },
            {
                "name": "Push Notifications",
                "key": "push",
                "configured": bool(cfg.push_provider and cfg.push_token),
                "configure_url": "/settings?section=integrations",
            },
        ]
        return jsonify({"integrations": integrations}), 200

    @bp.route("/api/calendar/events", methods=["GET"])
    def _calendar_events() -> Any:
        """Return calendar events for the resolved requesting user only.

        Identity comes from an explicit ``?user=`` query parameter or the
        standard identity chain; without a valid user the request fails
        closed (no global-credential fallback for named users).
        """
        return _calendar_events_response()

    @bp.route("/api/email/inbox", methods=["GET"])
    def _email_inbox() -> Any:
        """Return inbox messages for the resolved requesting user only.

        Identity comes from an explicit ``?user=`` query parameter or the
        standard identity chain; without a valid user the request fails
        closed (no global-credential fallback for named users).
        """
        from flask import jsonify, request

        from rex.identity import resolve_active_user
        from rex.integrations.email_service import create_email_service_for_user

        no_user_error = (
            "No active user for email. Set one with 'rex identify --user <id>' "
            "or pass an explicit ?user= parameter."
        )

        try:
            from rex.config_manager import load_config as _load_raw_config

            raw_config = _load_raw_config()
        except Exception:
            raw_config = {}

        explicit = str(request.args.get("user") or "").strip() or None
        try:
            user_id = resolve_active_user(explicit, config=raw_config)
        except ValueError:
            user_id = None
        if not user_id:
            return (
                jsonify({"ok": False, "messages": [], "configured": False, "error": no_user_error}),
                403,
            )

        try:
            svc, provider = create_email_service_for_user(user_id, raw_config)
        except PermissionError:
            return (
                jsonify({"ok": False, "messages": [], "configured": False, "error": no_user_error}),
                403,
            )

        if provider == "outlook":
            return (
                jsonify(
                    {
                        "ok": False,
                        "messages": [],
                        "configured": True,
                        "error": (
                            "Outlook email sync is not implemented yet. The current Outlook "
                            "settings only store app credentials; Rex cannot read Outlook mail "
                            "until Microsoft Graph OAuth token support is added."
                        ),
                    }
                ),
                501,
            )

        if svc is None:
            return jsonify({"ok": True, "messages": [], "configured": False}), 200

        try:
            limit = int(request.args.get("limit", 50))
        except ValueError:
            limit = 50

        messages = svc.list_inbox(limit=limit)
        configured = True

        def _msg_to_dict(m: Any) -> dict:
            return {
                "id": m.id,
                "thread_id": m.thread_id,
                "subject": m.subject,
                "sender": m.sender,
                "recipients": list(m.recipients),
                "body_text": m.body_text,
                "received_at": m.received_at.isoformat(),
                "labels": list(m.labels),
                "is_read": m.is_read,
                "priority": m.priority,
            }

        return (
            jsonify(
                {
                    "ok": True,
                    "messages": [_msg_to_dict(m) for m in messages],
                    "configured": configured,
                }
            ),
            200,
        )

    @bp.route("/api/sms/threads", methods=["GET"])
    def _sms_threads() -> Any:
        """Return SMS threads from the configured provider."""
        from flask import jsonify

        from rex.integrations.sms_service import SMSService

        sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        token = os.getenv("TWILIO_AUTH_TOKEN", "")
        provider = "twilio" if (sid and token) else "none"
        svc = SMSService(sms_provider=provider)

        threads = svc.list_threads()
        configured = provider != "none"

        def _msg_to_dict(m: Any) -> dict:
            return {
                "id": m.id,
                "thread_id": m.thread_id,
                "direction": m.direction,
                "body": m.body,
                "from_number": m.from_number,
                "to_number": m.to_number,
                "sent_at": m.sent_at.isoformat(),
                "status": m.status,
            }

        def _thread_to_dict(t: Any) -> dict:
            return {
                "id": t.id,
                "contact_name": t.contact_name,
                "contact_number": t.contact_number,
                "messages": [_msg_to_dict(m) for m in t.messages],
                "last_message_at": t.last_message_at.isoformat(),
                "unread_count": t.unread_count,
            }

        return (
            jsonify(
                {
                    "ok": True,
                    "threads": [_thread_to_dict(t) for t in threads],
                    "configured": configured,
                }
            ),
            200,
        )

    # ------------------------------------------------------------------
    # Capabilities and tools (public/auth)
    # ------------------------------------------------------------------

    @bp.route("/api/capabilities", methods=["GET"])
    def _list_capabilities() -> Any:
        """Return all capabilities from the capability registry (public)."""
        from flask import jsonify

        try:
            from rex.capabilities.registry import get_capability_registry

            registry = get_capability_registry()
            caps = [
                {
                    "name": c.name,
                    "description": c.description,
                    "category": getattr(c, "category", "General"),
                    "enabled": c.enabled,
                }
                for c in registry.list()
            ]
        except Exception:
            caps = []
        return jsonify({"capabilities": caps}), 200

    @bp.route("/api/tools", methods=["GET"])
    def _list_tools() -> Any:
        """Return registered tools with health status."""
        from flask import jsonify

        from rex.routes._helpers import _require_auth

        _, err = _require_auth()
        if err is not None:
            return err
        try:
            from rex.openclaw.tool_registry import get_tool_registry

            registry = get_tool_registry()
            tool_list = registry.list_tools(include_disabled=True)
            tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "capabilities": t.capabilities,
                    "enabled": t.enabled,
                    "version": t.version,
                }
                for t in tool_list
            ]
        except Exception:
            tools = []
        return jsonify({"tools": tools}), 200

    return bp
