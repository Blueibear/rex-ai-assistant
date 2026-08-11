from __future__ import annotations

import asyncio

import pytest

from rex.runtime.cancellation import TurnCancelledError, current_turn_cancellation
from rex.runtime.events import EventKind
from rex.runtime.turn import (
    AuthorizationSnapshotRef,
    ResponseMode,
    TurnContext,
    TurnScope,
    TurnSource,
)
from rex.runtime.turn_engine import TurnEngine


def _context(user_id: str) -> TurnContext:
    return TurnContext.create(
        user_id=user_id,
        scope=TurnScope.USER,
        source=TurnSource.MOBILE,
        device_id=f"{user_id}-phone",
        response_mode=ResponseMode.SCREEN,
        authorization=AuthorizationSnapshotRef("policy:v1", f"permissions:{user_id}:v1"),
    )


@pytest.mark.asyncio
async def test_cancelling_one_concurrent_turn_does_not_cancel_the_other() -> None:
    engine = TurnEngine()
    james = _context("james")
    cole = _context("cole")
    james_events = []
    cole_events = []
    started = asyncio.Event()

    async def james_work(_events):
        assert current_turn_cancellation() is james.cancellation
        started.set()
        await asyncio.sleep(0)
        james.cancellation.cancel("replacement_turn")
        james.cancellation.raise_if_cancelled()

    async def cole_work(_events):
        assert current_turn_cancellation() is cole.cancellation
        await started.wait()
        await asyncio.sleep(0)
        assert cole.cancellation.cancelled is False
        return "cole-result"

    james_task = asyncio.create_task(
        engine.execute_async(james, james_work, on_event=james_events.append)
    )
    cole_task = asyncio.create_task(
        engine.execute_async(cole, cole_work, on_event=cole_events.append)
    )

    james_result, cole_result = await asyncio.gather(james_task, cole_task, return_exceptions=True)

    assert isinstance(james_result, TurnCancelledError)
    assert cole_result == "cole-result"
    assert james_events[-1].kind is EventKind.CANCELLED
    assert cole_events[-1].kind is EventKind.COMPLETED
    assert {event.user_id for event in james_events} == {"james"}
    assert {event.user_id for event in cole_events} == {"cole"}
    assert current_turn_cancellation() is None
