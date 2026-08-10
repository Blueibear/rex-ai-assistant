from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from importlib.util import find_spec
from itertools import count

import pytest

from rex.runtime.events import EventKind, TerminalStateError, TurnEventStream
from rex.runtime.turn import (
    AuthorizationSnapshotRef,
    ResponseMode,
    TurnContext,
    TurnScope,
    TurnSource,
)
from rex.runtime.turn_engine import TurnEngine


def _context(user_id: str = "james") -> TurnContext:
    return TurnContext.create(
        user_id=user_id,
        scope=TurnScope.USER,
        source=TurnSource.ELECTRON,
        device_id="desktop-1",
        response_mode=ResponseMode.SCREEN,
        authorization=AuthorizationSnapshotRef(
            policy_ref="policy:v1",
            permission_ref=f"permissions:{user_id}:v1",
        ),
    )


def test_turn_runtime_modules_exist() -> None:
    assert find_spec("rex.runtime.turn") is not None
    assert find_spec("rex.runtime.events") is not None
    assert find_spec("rex.runtime.turn_engine") is not None


def test_turn_context_validates_identity_and_is_immutable() -> None:
    context = TurnContext.create(
        user_id="james",
        scope=TurnScope.USER,
        source=TurnSource.VOICE,
        device_id="kitchen-mic",
        response_mode=ResponseMode.VOICE,
        authorization=AuthorizationSnapshotRef("policy:v2", "permissions:james:v5"),
        timeout_seconds=2.0,
        clock=lambda: 1_000,
    )

    assert context.user_id == "james"
    assert context.started_monotonic_ns == 1_000
    assert context.deadline_monotonic_ns == 2_000_001_000
    with pytest.raises(FrozenInstanceError):
        context.user_id = "cole"  # type: ignore[misc]
    with pytest.raises(ValueError, match="Invalid user_id"):
        _context("../cole")


def test_turn_ids_are_unique() -> None:
    assert _context().turn_id != _context().turn_id


def test_event_stream_orders_and_correlates_all_progress_stages() -> None:
    ticks = count(100, 10)
    context = _context()
    events = TurnEventStream(context, clock=lambda: next(ticks))
    kinds = [
        EventKind.TURN_STARTED,
        EventKind.CONTEXT_PROGRESS,
        EventKind.ROUTE_PROGRESS,
        EventKind.CAPABILITY_PROGRESS,
        EventKind.ACTION_PROGRESS,
        EventKind.MODEL_PROGRESS,
        EventKind.RESPONSE_PROGRESS,
    ]
    emitted = [events.emit(kind) for kind in kinds]
    emitted.append(events.finish(EventKind.COMPLETED))

    assert [event.sequence for event in emitted] == list(range(1, 9))
    assert [event.monotonic_ns for event in emitted] == list(range(100, 180, 10))
    assert all(event.turn_id == context.turn_id for event in emitted)
    assert all(event.user_id == "james" for event in emitted)
    assert emitted[-1].is_terminal is True


def test_terminal_state_fails_closed() -> None:
    events = TurnEventStream(_context())
    events.emit(EventKind.TURN_STARTED)
    events.finish(EventKind.CANCELLED, {"reason": "caller_cancelled"})

    with pytest.raises(TerminalStateError):
        events.finish(EventKind.COMPLETED)
    with pytest.raises(TerminalStateError):
        events.emit(EventKind.RESPONSE_PROGRESS)


def test_turn_engine_preserves_success_value_and_emits_one_terminal() -> None:
    observed = []

    def operation(events: TurnEventStream) -> str:
        events.emit(EventKind.ROUTE_PROGRESS, {"route": "direct"})
        return "same-public-result"

    result = TurnEngine().execute(_context(), operation, on_event=observed.append)

    assert result == "same-public-result"
    assert [event.kind for event in observed] == [
        EventKind.TURN_STARTED,
        EventKind.ROUTE_PROGRESS,
        EventKind.COMPLETED,
    ]
    assert sum(event.is_terminal for event in observed) == 1


def test_turn_engine_preserves_failure_and_emits_failed_terminal() -> None:
    observed = []

    class ExpectedFailure(RuntimeError):
        pass

    def operation(events: TurnEventStream) -> str:
        events.emit(EventKind.MODEL_PROGRESS)
        raise ExpectedFailure("provider failed")

    with pytest.raises(ExpectedFailure, match="provider failed"):
        TurnEngine().execute(_context(), operation, on_event=observed.append)

    assert observed[-1].kind is EventKind.FAILED
    assert sum(event.is_terminal for event in observed) == 1


def test_concurrent_users_have_isolated_event_correlation() -> None:
    engine = TurnEngine()

    def run(user_id: str) -> tuple[TurnContext, list]:
        context = _context(user_id)
        observed = []
        engine.execute(
            context,
            lambda stream: stream.emit(EventKind.CONTEXT_PROGRESS),
            on_event=observed.append,
        )
        return context, observed

    with ThreadPoolExecutor(max_workers=2) as pool:
        james_future = pool.submit(run, "james")
        cole_future = pool.submit(run, "cole")
        james_context, james_events = james_future.result()
        cole_context, cole_events = cole_future.result()

    assert james_context.turn_id != cole_context.turn_id
    assert {event.user_id for event in james_events} == {"james"}
    assert {event.user_id for event in cole_events} == {"cole"}
    assert {event.turn_id for event in james_events} == {james_context.turn_id}
    assert {event.turn_id for event in cole_events} == {cole_context.turn_id}


def test_turn_context_rejects_invalid_scope_on_direct_construction() -> None:
    with pytest.raises(ValueError, match="turn scope"):
        replace(_context(), scope="global")  # type: ignore[arg-type]


def test_event_observer_failure_does_not_change_wrapped_result() -> None:
    def broken_observer(_event: object) -> None:
        raise RuntimeError("observer unavailable")

    result = TurnEngine().execute(
        _context(),
        lambda _events: "operation-result",
        on_event=broken_observer,
    )

    assert result == "operation-result"
