"""Tests for US-087: Multi-channel message router.

Acceptance criteria:
- router accepts a message payload and a target channel identifier
- routes correctly to dashboard, email, and SMS backends based on channel value
- unknown or unconfigured channel raises a handled error and does not crash
- active channel configurable without code changes
- Typecheck passes
"""

from __future__ import annotations

import pytest

from rex.dashboard.sse import NotificationBroadcaster
from rex.email_backends.stub import StubEmailBackend
from rex.messaging_backends.message_router import (
    CHANNEL_DASHBOARD,
    CHANNEL_EMAIL,
    CHANNEL_SMS,
    KNOWN_CHANNELS,
    ChannelNotConfiguredError,
    MessagePayload,
    MessageRouter,
    RouterConfig,
    RouteResult,
    UnknownChannelError,
)
from rex.messaging_backends.sms_sender_stub import SmsSenderStub

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_sms_stub() -> SmsSenderStub:
    return SmsSenderStub()


def make_email_stub() -> StubEmailBackend:
    backend = StubEmailBackend(fixture_path=None)
    # Override send so it works without a fixture file
    return backend


def make_broadcaster() -> NotificationBroadcaster:
    return NotificationBroadcaster()


def make_full_router(
    active: str = CHANNEL_DASHBOARD,
) -> tuple[MessageRouter, SmsSenderStub, StubEmailBackend, NotificationBroadcaster]:
    sms = make_sms_stub()
    email = make_email_stub()
    broadcaster = make_broadcaster()
    router = MessageRouter(
        config=RouterConfig(active_channel=active),
        sms_backend=sms,
        email_backend=email,
        dashboard_broadcaster=broadcaster,
    )
    return router, sms, email, broadcaster


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


class TestMessagePayload:
    def test_minimal_payload(self) -> None:
        p = MessagePayload(body="Hello")
        assert p.body == "Hello"
        assert p.subject == ""
        assert p.to == ""
        assert p.metadata == {}

    def test_full_payload(self) -> None:
        p = MessagePayload(
            body="Meeting at 3pm",
            subject="Reminder",
            to="+15555550100",
            metadata={"priority": "high"},
        )
        assert p.body == "Meeting at 3pm"
        assert p.subject == "Reminder"
        assert p.to == "+15555550100"
        assert p.metadata["priority"] == "high"


# ---------------------------------------------------------------------------
# KNOWN_CHANNELS constant
# ---------------------------------------------------------------------------


class TestKnownChannels:
    def test_known_channels_contains_all_three(self) -> None:
        assert CHANNEL_DASHBOARD in KNOWN_CHANNELS
        assert CHANNEL_EMAIL in KNOWN_CHANNELS
        assert CHANNEL_SMS in KNOWN_CHANNELS

    def test_known_channels_is_frozenset(self) -> None:
        assert isinstance(KNOWN_CHANNELS, frozenset)


# ---------------------------------------------------------------------------
# RouterConfig
# ---------------------------------------------------------------------------


class TestRouterConfig:
    def test_default_active_channel_is_dashboard(self) -> None:
        cfg = RouterConfig()
        assert cfg.active_channel == CHANNEL_DASHBOARD

    def test_custom_active_channel(self) -> None:
        cfg = RouterConfig(active_channel=CHANNEL_SMS)
        assert cfg.active_channel == CHANNEL_SMS


# ---------------------------------------------------------------------------
# SMS routing
# ---------------------------------------------------------------------------


class TestSmsRouting:
    def test_route_to_sms_succeeds(self) -> None:
        router, sms, _, _ = make_full_router()
        payload = MessagePayload(body="Test SMS", to="+15555550100")
        result = router.route(payload, CHANNEL_SMS)
        assert isinstance(result, RouteResult)
        assert result.ok is True
        assert result.channel == CHANNEL_SMS

    def test_sms_message_appears_in_stub_log(self) -> None:
        router, sms, _, _ = make_full_router()
        payload = MessagePayload(body="Hello SMS", to="+15555550199")
        router.route(payload, CHANNEL_SMS)
        assert len(sms.sent_messages) == 1
        assert sms.sent_messages[0].to == "+15555550199"
        assert sms.sent_messages[0].body == "Hello SMS"

    def test_route_sms_detail_contains_ok(self) -> None:
        router, _, _, _ = make_full_router()
        result = router.route(MessagePayload(body="Hi", to="+15555550100"), CHANNEL_SMS)
        assert isinstance(result.detail, dict)
        assert result.detail.get("ok") is True

    def test_multiple_sms_messages_logged(self) -> None:
        router, sms, _, _ = make_full_router()
        for i in range(3):
            router.route(MessagePayload(body=f"msg{i}", to="+15555550100"), CHANNEL_SMS)
        assert len(sms.sent_messages) == 3


# ---------------------------------------------------------------------------
# Email routing
# ---------------------------------------------------------------------------


class TestEmailRouting:
    def test_route_to_email_succeeds(self) -> None:
        router, _, email, _ = make_full_router()
        payload = MessagePayload(
            body="Hello email",
            subject="Test subject",
            to="user@example.com",
        )
        result = router.route(payload, CHANNEL_EMAIL)
        assert result.ok is True
        assert result.channel == CHANNEL_EMAIL

    def test_email_body_forwarded(self) -> None:
        router, _, email, _ = make_full_router()
        payload = MessagePayload(
            body="Important content",
            subject="Subj",
            to="a@b.com",
        )
        router.route(payload, CHANNEL_EMAIL)
        sent = email.sent_messages
        assert len(sent) == 1
        assert sent[0]["body"] == "Important content"
        assert sent[0]["subject"] == "Subj"

    def test_email_subject_defaults_when_empty(self) -> None:
        router, _, email, _ = make_full_router()
        payload = MessagePayload(body="No subject", to="x@y.com")
        router.route(payload, CHANNEL_EMAIL)
        sent = email.sent_messages
        assert sent[0]["subject"] == "(no subject)"

    def test_email_to_from_metadata(self) -> None:
        router, _, email, _ = make_full_router()
        payload = MessagePayload(
            body="Via metadata",
            metadata={"to_addrs": ["one@a.com", "two@b.com"], "from_addr": "rex@local"},
        )
        router.route(payload, CHANNEL_EMAIL)
        sent = email.sent_messages
        assert sent[0]["to_addrs"] == ["one@a.com", "two@b.com"]
        assert sent[0]["from_addr"] == "rex@local"


# ---------------------------------------------------------------------------
# Dashboard routing
# ---------------------------------------------------------------------------


class TestDashboardRouting:
    def test_route_to_dashboard_succeeds(self) -> None:
        router, _, _, broadcaster = make_full_router(CHANNEL_DASHBOARD)
        subscriber = broadcaster.subscribe()
        payload = MessagePayload(body="Dashboard message", subject="Alert")
        result = router.route(payload, CHANNEL_DASHBOARD)
        assert result.ok is True
        assert result.channel == CHANNEL_DASHBOARD
        broadcaster.unsubscribe(subscriber)

    def test_dashboard_event_received_by_subscriber(self) -> None:
        router, _, _, broadcaster = make_full_router()
        subscriber = broadcaster.subscribe()
        payload = MessagePayload(body="Ping", subject="Test")
        router.route(payload, CHANNEL_DASHBOARD)
        event = subscriber.queue.get_nowait()
        assert event["type"] == "message"
        assert event["body"] == "Ping"
        broadcaster.unsubscribe(subscriber)

    def test_dashboard_detail_contains_body(self) -> None:
        router, _, _, _ = make_full_router()
        result = router.route(MessagePayload(body="Check detail"), CHANNEL_DASHBOARD)
        assert isinstance(result.detail, dict)
        assert result.detail["body"] == "Check detail"

    def test_dashboard_metadata_forwarded(self) -> None:
        router, _, _, broadcaster = make_full_router()
        subscriber = broadcaster.subscribe()
        payload = MessagePayload(body="meta msg", metadata={"priority": "high"})
        router.route(payload, CHANNEL_DASHBOARD)
        event = subscriber.queue.get_nowait()
        assert event.get("priority") == "high"
        broadcaster.unsubscribe(subscriber)


# ---------------------------------------------------------------------------
# Unknown channel — handled error, does not crash
# ---------------------------------------------------------------------------


class TestUnknownChannel:
    def test_unknown_channel_raises_unknown_channel_error(self) -> None:
        router, _, _, _ = make_full_router()
        with pytest.raises(UnknownChannelError) as exc_info:
            router.route(MessagePayload(body="x"), "pigeon")
        assert exc_info.value.channel == "pigeon"

    def test_unknown_channel_error_message_lists_supported(self) -> None:
        router, _, _, _ = make_full_router()
        with pytest.raises(UnknownChannelError) as exc_info:
            router.route(MessagePayload(body="x"), "fax")
        assert "fax" in str(exc_info.value)

    def test_unknown_channel_does_not_crash_assistant(self) -> None:
        """Demonstrate that callers can handle the error without crashing."""
        router, _, _, _ = make_full_router()
        try:
            router.route(MessagePayload(body="x"), "unknown_channel")
            crashed = False
        except UnknownChannelError:
            crashed = False  # handled error — not a crash
        except Exception:
            crashed = True
        assert not crashed

    def test_empty_channel_string_raises_unknown_channel_error(self) -> None:
        router, _, _, _ = make_full_router()
        with pytest.raises(UnknownChannelError):
            router.route(MessagePayload(body="x"), "")


# ---------------------------------------------------------------------------
# Unconfigured (None) backend — handled error
# ---------------------------------------------------------------------------


class TestUnconfiguredBackend:
    def test_sms_not_configured_raises_channel_not_configured_error(self) -> None:
        router = MessageRouter(config=RouterConfig(active_channel=CHANNEL_SMS))
        with pytest.raises(ChannelNotConfiguredError) as exc_info:
            router.route(MessagePayload(body="x"), CHANNEL_SMS)
        assert exc_info.value.channel == CHANNEL_SMS

    def test_email_not_configured_raises_channel_not_configured_error(self) -> None:
        router = MessageRouter(config=RouterConfig(active_channel=CHANNEL_EMAIL))
        with pytest.raises(ChannelNotConfiguredError) as exc_info:
            router.route(MessagePayload(body="x"), CHANNEL_EMAIL)
        assert exc_info.value.channel == CHANNEL_EMAIL

    def test_dashboard_not_configured_raises_channel_not_configured_error(self) -> None:
        router = MessageRouter(config=RouterConfig(active_channel=CHANNEL_DASHBOARD))
        with pytest.raises(ChannelNotConfiguredError) as exc_info:
            router.route(MessagePayload(body="x"), CHANNEL_DASHBOARD)
        assert exc_info.value.channel == CHANNEL_DASHBOARD


# ---------------------------------------------------------------------------
# Active channel configurability (no code changes required)
# ---------------------------------------------------------------------------


class TestActiveChannelConfig:
    def test_default_router_active_channel_is_configurable(self) -> None:
        router, sms, _, _ = make_full_router(CHANNEL_SMS)
        assert router.active_channel == CHANNEL_SMS

    def test_send_uses_active_channel(self) -> None:
        router, sms, _, _ = make_full_router(CHANNEL_SMS)
        payload = MessagePayload(body="via active channel", to="+15555550100")
        result = router.send(payload)
        assert result.channel == CHANNEL_SMS
        assert result.ok is True

    def test_active_channel_can_be_changed_at_runtime(self) -> None:
        router, sms, _, broadcaster = make_full_router(CHANNEL_SMS)
        router.active_channel = CHANNEL_DASHBOARD
        assert router.active_channel == CHANNEL_DASHBOARD

    def test_configure_replaces_config(self) -> None:
        router, _, _, _ = make_full_router(CHANNEL_DASHBOARD)
        router.configure(RouterConfig(active_channel=CHANNEL_EMAIL))
        assert router.active_channel == CHANNEL_EMAIL

    def test_setting_unknown_active_channel_raises(self) -> None:
        router, _, _, _ = make_full_router()
        with pytest.raises(UnknownChannelError):
            router.active_channel = "morse_code"

    def test_routing_via_different_active_channels(self) -> None:
        """Same router, different active channel, different backend receives message."""
        sms = make_sms_stub()
        broadcaster = make_broadcaster()
        email = make_email_stub()

        router = MessageRouter(
            config=RouterConfig(active_channel=CHANNEL_SMS),
            sms_backend=sms,
            email_backend=email,
            dashboard_broadcaster=broadcaster,
        )
        sub = broadcaster.subscribe()

        router.send(MessagePayload(body="sms msg", to="+15550001111"))
        assert len(sms.sent_messages) == 1
        assert sub.queue.empty()

        router.active_channel = CHANNEL_DASHBOARD
        router.send(MessagePayload(body="dash msg"))
        assert len(sms.sent_messages) == 1  # unchanged
        assert not sub.queue.empty()

        broadcaster.unsubscribe(sub)

    def test_router_config_loaded_from_dict(self) -> None:
        """Simulates loading config from an external source (e.g. JSON)."""
        raw = {"active_channel": "email"}
        cfg = RouterConfig(**raw)
        router = MessageRouter(config=cfg, email_backend=make_email_stub())
        payload = MessagePayload(body="from config", to="a@b.com")
        result = router.send(payload)
        assert result.channel == CHANNEL_EMAIL


# ---------------------------------------------------------------------------
# RouteResult structure
# ---------------------------------------------------------------------------


class TestRouteResult:
    def test_successful_route_result_ok_true(self) -> None:
        router, _, _, _ = make_full_router()
        result = router.route(MessagePayload(body="ping"), CHANNEL_DASHBOARD)
        assert result.ok is True
        assert result.error is None

    def test_route_result_has_channel_set(self) -> None:
        router, sms, _, _ = make_full_router()
        result = router.route(MessagePayload(body="x", to="+15550000001"), CHANNEL_SMS)
        assert result.channel == CHANNEL_SMS
