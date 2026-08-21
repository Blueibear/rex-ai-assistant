from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from rex.media.models import AudioTarget, MediaCapability, TargetKind
from rex.media.registry import AudioTargetRegistry
from rex.output_routing import (
    FallbackMode,
    OutputKind,
    OutputRoutingService,
    QuietHours,
    RoutingRule,
    UserOutputPolicy,
)


def _target(
    target_id: str,
    name: str,
    *,
    room: str,
    online: bool = True,
) -> AudioTarget:
    return AudioTarget(
        id=target_id,
        native_id=target_id,
        provider="test",
        kind=TargetKind.SPEAKER,
        display_name=name,
        aliases=(),
        room=room,
        capabilities=frozenset({MediaCapability.PLAY, MediaCapability.SET_VOLUME}),
        online=online,
        health="online" if online else "offline",
    )


def _service(tmp_path) -> OutputRoutingService:
    registry = AudioTargetRegistry(
        (
            _target("test:bedroom", "Bedroom Speaker", room="bedroom"),
            _target("test:kitchen", "Kitchen Speaker", room="kitchen"),
            _target("test:office", "Office Speaker", room="office"),
            _target(
                "test:offline",
                "Offline Speaker",
                room="garage",
                online=False,
            ),
        ),
        authorized_target_ids={
            "james": {
                "test:bedroom",
                "test:kitchen",
                "test:office",
                "test:offline",
            },
            "cole": {"test:kitchen"},
        },
        origin_device_targets={
            "james-phone": "test:office",
            "cole-phone": "test:kitchen",
        },
    )
    return OutputRoutingService(registry, root=tmp_path)


def _at(hour: int, *, weekday_offset: int = 0) -> datetime:
    # 2026-08-17 is Monday.
    return datetime(
        2026,
        8,
        17 + weekday_offset,
        hour,
        0,
        tzinfo=ZoneInfo("America/Chicago"),
    )


def test_policy_round_trips_in_user_private_partition(tmp_path) -> None:
    service = _service(tmp_path)
    policy = UserOutputPolicy(
        spoken_response_target_id="test:office",
        timer_target_id="test:kitchen",
        alarm_target_id="test:bedroom",
        media_target_id="test:kitchen",
        media_fallback=FallbackMode.NAMED,
        media_fallback_target_id="test:office",
        media_volume=35,
        default_media_provider="apple_music",
        default_media_account_id="james-main",
        quiet_hours=QuietHours(
            enabled=True,
            start_local_time=time(23, 0),
            end_local_time=time(6, 0),
            days_of_week=(0, 1, 2, 3, 4),
        ),
        rules=(
            RoutingRule(
                output_kind=OutputKind.ALARM,
                target_id="test:bedroom",
                days_of_week=(0, 1, 2, 3, 4),
                start_local_time=time(5, 0),
                end_local_time=time(10, 0),
                target_volume=42,
            ),
        ),
    )

    service.save_policy("james", policy)

    assert service.get_policy("james") == policy
    assert service.get_policy("cole") == UserOutputPolicy()
    assert (
        tmp_path / "users" / "james" / "output_routing" / "policy.json"
    ).is_file()
    assert not (
        tmp_path / "users" / "cole" / "output_routing" / "policy.json"
    ).exists()


def test_explicit_target_beats_request_origin_and_default(tmp_path) -> None:
    service = _service(tmp_path)
    service.save_policy(
        "james",
        UserOutputPolicy(media_target_id="test:kitchen"),
    )

    route = service.resolve(
        user_id="james",
        output_kind=OutputKind.MEDIA,
        explicit_target_text="bedroom",
        origin_device_id="james-phone",
        at=_at(12),
    )

    assert route.target_id == "test:bedroom"
    assert route.reason == "explicit_target"


def test_interactive_media_prefers_authorized_request_origin(tmp_path) -> None:
    service = _service(tmp_path)
    service.save_policy(
        "james",
        UserOutputPolicy(media_target_id="test:kitchen", media_volume=28),
    )

    route = service.resolve(
        user_id="james",
        output_kind=OutputKind.MEDIA,
        explicit_target_text=None,
        origin_device_id="james-phone",
        at=_at(12),
    )

    assert route.target_id == "test:office"
    assert route.reason == "request_origin"
    assert route.target_volume == 28


def test_conditional_rule_beats_default(tmp_path) -> None:
    service = _service(tmp_path)
    service.save_policy(
        "james",
        UserOutputPolicy(
            alarm_target_id="test:kitchen",
            alarm_volume=50,
            rules=(
                RoutingRule(
                    output_kind=OutputKind.ALARM,
                    target_id="test:bedroom",
                    days_of_week=(0,),
                    start_local_time=time(5),
                    end_local_time=time(9),
                    target_volume=30,
                ),
            ),
        ),
    )

    route = service.resolve(
        user_id="james",
        output_kind=OutputKind.ALARM,
        explicit_target_text=None,
        origin_device_id=None,
        at=_at(7),
    )

    assert route.target_id == "test:bedroom"
    assert route.reason == "conditional_rule"
    assert route.rule_index == 0
    assert route.target_volume == 30


def test_named_fallback_is_explicit_in_decision(tmp_path) -> None:
    service = _service(tmp_path)
    service.save_policy(
        "james",
        UserOutputPolicy(
            timer_target_id="test:offline",
            timer_fallback=FallbackMode.NAMED,
            timer_fallback_target_id="test:kitchen",
            timer_volume=40,
        ),
    )

    route = service.resolve(
        user_id="james",
        output_kind=OutputKind.TIMER,
        explicit_target_text=None,
        origin_device_id=None,
        at=_at(12),
    )

    assert route.target_id == "test:kitchen"
    assert route.reason == "named_fallback"
    assert route.fallback_mode is FallbackMode.NAMED
    assert route.fallback_from == "test:offline"
    assert route.target_volume == 40


def test_ask_fallback_never_silently_reroutes(tmp_path) -> None:
    service = _service(tmp_path)
    service.save_policy(
        "james",
        UserOutputPolicy(
            alarm_target_id="test:offline",
            alarm_fallback=FallbackMode.ASK,
        ),
    )

    route = service.resolve(
        user_id="james",
        output_kind=OutputKind.ALARM,
        explicit_target_text=None,
        origin_device_id=None,
        at=_at(12),
    )

    assert route.target_id is None
    assert route.reason == "fallback_confirmation_required"
    assert route.requires_confirmation is True


def test_no_fallback_reports_unavailable_target(tmp_path) -> None:
    service = _service(tmp_path)
    service.save_policy(
        "james",
        UserOutputPolicy(
            spoken_response_target_id="test:offline",
            spoken_response_fallback=FallbackMode.NONE,
        ),
    )

    route = service.resolve(
        user_id="james",
        output_kind=OutputKind.SPOKEN_RESPONSE,
        explicit_target_text=None,
        origin_device_id=None,
        at=_at(12),
    )

    assert route.target_id is None
    assert route.reason == "configured_target_unavailable"
    assert route.fallback_mode is FallbackMode.NONE


def test_quiet_hours_suppress_optional_audio_but_not_alarm(tmp_path) -> None:
    service = _service(tmp_path)
    quiet = QuietHours(
        enabled=True,
        start_local_time=time(22),
        end_local_time=time(7),
    )
    service.save_policy(
        "james",
        UserOutputPolicy(
            spoken_response_target_id="test:office",
            alarm_target_id="test:bedroom",
            quiet_hours=quiet,
        ),
    )

    spoken = service.resolve(
        user_id="james",
        output_kind=OutputKind.SPOKEN_RESPONSE,
        explicit_target_text=None,
        origin_device_id=None,
        at=_at(23),
    )
    alarm = service.resolve(
        user_id="james",
        output_kind=OutputKind.ALARM,
        explicit_target_text=None,
        origin_device_id=None,
        at=_at(23),
    )

    assert spoken.suppressed is True
    assert spoken.reason == "quiet_hours"
    assert alarm.target_id == "test:bedroom"
    assert alarm.suppressed is False


def test_explicit_target_overrides_quiet_hours(tmp_path) -> None:
    service = _service(tmp_path)
    service.save_policy(
        "james",
        UserOutputPolicy(
            spoken_response_target_id="test:office",
            quiet_hours=QuietHours(
                enabled=True,
                start_local_time=time(22),
                end_local_time=time(7),
            ),
        ),
    )

    route = service.resolve(
        user_id="james",
        output_kind=OutputKind.SPOKEN_RESPONSE,
        explicit_target_text="kitchen",
        origin_device_id=None,
        at=_at(23),
    )

    assert route.target_id == "test:kitchen"
    assert route.reason == "explicit_target"
    assert route.suppressed is False


def test_policy_never_widens_target_authority(tmp_path) -> None:
    service = _service(tmp_path)
    service.save_policy(
        "cole",
        UserOutputPolicy(
            media_target_id="test:office",
            media_fallback=FallbackMode.NONE,
        ),
    )

    route = service.resolve(
        user_id="cole",
        output_kind=OutputKind.MEDIA,
        explicit_target_text=None,
        origin_device_id=None,
        at=_at(12),
    )

    assert route.target_id is None
    assert route.reason == "configured_target_unavailable"


def test_user_policies_are_isolated_under_concurrent_reads(tmp_path) -> None:
    service = _service(tmp_path)
    service.save_policy("james", UserOutputPolicy(timer_target_id="test:bedroom"))
    service.save_policy("cole", UserOutputPolicy(timer_target_id="test:kitchen"))

    james = service.resolve(
        user_id="james",
        output_kind=OutputKind.TIMER,
        explicit_target_text=None,
        origin_device_id=None,
        at=_at(12),
    )
    cole = service.resolve(
        user_id="cole",
        output_kind=OutputKind.TIMER,
        explicit_target_text=None,
        origin_device_id=None,
        at=_at(12),
    )

    assert james.target_id == "test:bedroom"
    assert cole.target_id == "test:kitchen"
