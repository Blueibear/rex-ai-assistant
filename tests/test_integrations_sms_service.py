"""Unit tests for rex.integrations.sms_service — stub mode."""

from __future__ import annotations

import pytest

from rex.integrations.models import SMSMessage, SMSThread
from rex.integrations.sms_service import SMSService


class TestSMSServiceStub:
    """Tests for SMSService running in stub mode (sms_provider='none')."""

    def setup_method(self) -> None:
        self.service = SMSService(sms_provider="none")

    # ------------------------------------------------------------------
    # list_threads
    # ------------------------------------------------------------------

    def test_list_threads_returns_list(self) -> None:
        threads = self.service.list_threads()
        assert isinstance(threads, list)
        assert len(threads) > 0

    def test_list_threads_returns_two_stubs(self) -> None:
        threads = self.service.list_threads()
        assert len(threads) == 2

    def test_list_threads_all_sms_thread(self) -> None:
        threads = self.service.list_threads()
        assert all(isinstance(t, SMSThread) for t in threads)

    def test_list_threads_each_has_messages(self) -> None:
        threads = self.service.list_threads()
        for t in threads:
            assert len(t.messages) > 0

    def test_list_threads_messages_are_sms_message(self) -> None:
        threads = self.service.list_threads()
        for t in threads:
            assert all(isinstance(m, SMSMessage) for m in t.messages)

    # ------------------------------------------------------------------
    # get_thread
    # ------------------------------------------------------------------

    def test_get_thread_returns_matching_thread(self) -> None:
        threads = self.service.list_threads()
        first = threads[0]
        found = self.service.get_thread(first.id)
        assert found.id == first.id

    def test_get_thread_raises_for_unknown_id(self) -> None:
        with pytest.raises(KeyError):
            self.service.get_thread("does-not-exist")

    # ------------------------------------------------------------------
    # send
    # ------------------------------------------------------------------

    def test_send_returns_sms_message(self) -> None:
        msg = self.service.send("+14155559999", "Hello from test")
        assert isinstance(msg, SMSMessage)

    def test_send_stub_status(self) -> None:
        msg = self.service.send("+14155559999", "Test body")
        assert msg.status == "stub"

    def test_send_direction_is_outbound(self) -> None:
        msg = self.service.send("+14155559999", "Test body")
        assert msg.direction == "outbound"

    def test_send_to_number_matches(self) -> None:
        msg = self.service.send("+14155559999", "Test body")
        assert msg.to_number == "+14155559999"

    def test_send_body_matches(self) -> None:
        msg = self.service.send("+14155559999", "Test body content")
        assert msg.body == "Test body content"

    def test_send_has_id(self) -> None:
        msg = self.service.send("+14155559999", "Test body")
        assert msg.id != ""

    # ------------------------------------------------------------------
    # Model round-trip
    # ------------------------------------------------------------------

    def test_thread_model_dump_round_trip(self) -> None:
        thread = self.service.list_threads()[0]
        restored = SMSThread(**thread.model_dump())
        assert restored == thread
