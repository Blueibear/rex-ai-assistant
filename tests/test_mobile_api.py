"""Focused US-097 mobile TurnEngine adapter provenance tests."""

from __future__ import annotations

from rex.mobile_api.chat import MobileChatService
from rex.runtime.turn import TurnSource


class _CapturingAssistant:
    def __init__(self) -> None:
        self.invocations = []

    async def generate_reply(self, _message: str, **_kwargs) -> str:
        from rex.runtime.invocation import current_turn_invocation

        self.invocations.append(current_turn_invocation())
        return "ok"

    async def stream_reply(self, _message: str, **_kwargs):
        from rex.runtime.invocation import current_turn_invocation

        self.invocations.append(current_turn_invocation())
        yield "one"
        yield "two"


def test_mobile_generate_stamps_authenticated_device_provenance() -> None:
    assistant = _CapturingAssistant()
    service = MobileChatService(lambda: assistant)
    assert service.generate("hi", user_id="james", device_id="phone-123") == "ok"
    invocation = assistant.invocations[-1]
    assert invocation.source is TurnSource.MOBILE
    assert invocation.device_id == "phone-123"


def test_mobile_stream_stamps_same_authenticated_device_provenance() -> None:
    assistant = _CapturingAssistant()
    service = MobileChatService(lambda: assistant)
    chunks = list(service.stream("hi", user_id="james", device_id="phone-456"))
    assert chunks == ["one", "two"]
    invocation = assistant.invocations[-1]
    assert invocation.source is TurnSource.MOBILE
    assert invocation.device_id == "phone-456"


def test_mobile_adapter_does_not_fabricate_device_id() -> None:
    assistant = _CapturingAssistant()
    service = MobileChatService(lambda: assistant)
    assert service.generate("hi", user_id="james", device_id=None) == "ok"
    invocation = assistant.invocations[-1]
    assert invocation.source is TurnSource.MOBILE
    assert invocation.device_id is None


def test_mobile_stream_does_not_leak_provenance_between_yields() -> None:
    from rex.mobile_api.action_context import mobile_action_context_active
    from rex.runtime.invocation import current_turn_invocation

    assistant = _CapturingAssistant()
    service = MobileChatService(lambda: assistant)
    iterator = service.stream("hi", user_id="james", device_id="phone-789")
    assert next(iterator) == "one"
    assert current_turn_invocation().source is TurnSource.ASSISTANT
    assert current_turn_invocation().device_id is None
    assert mobile_action_context_active() is False
    iterator.close()
