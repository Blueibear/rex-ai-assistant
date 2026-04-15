"""Tests for US-PH-003: Outbound calling via Twilio REST API."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest import mock

from rex.telephony.outbound import (
    detect_call_intent,
    is_phone_number,
    log_call,
    lookup_contact,
    make_call,
)


def _make_twilio_mock() -> tuple[mock.MagicMock, mock.MagicMock]:
    """Return (mock_client_instance, mock_Client_class) for sys.modules injection."""
    mock_call = mock.MagicMock()
    mock_call.sid = "CA_test_sid"
    mock_client = mock.MagicMock()
    mock_client.calls.create.return_value = mock_call

    mock_Client_class = mock.MagicMock(return_value=mock_client)

    twilio_mod = mock.MagicMock()
    twilio_rest_mod = mock.MagicMock()
    twilio_rest_mod.Client = mock_Client_class

    return mock_client, mock_Client_class, twilio_mod, twilio_rest_mod


def _patch_twilio_modules(mock_client, mock_Client_class, twilio_mod, twilio_rest_mod):
    """Return a context manager that injects fake twilio modules into sys.modules."""
    return mock.patch.dict(
        sys.modules,
        {
            "twilio": twilio_mod,
            "twilio.rest": twilio_rest_mod,
            "twilio.twiml": mock.MagicMock(),
            "twilio.twiml.voice_response": mock.MagicMock(),
        },
    )


# ---------------------------------------------------------------------------
# detect_call_intent
# ---------------------------------------------------------------------------


class TestDetectCallIntent:
    def test_call_contact_name(self):
        target, msg = detect_call_intent("Call John Smith")
        assert target == "John Smith"
        assert msg is None

    def test_call_phone_number(self):
        target, msg = detect_call_intent("Call +15551234567")
        assert target == "+15551234567"
        assert msg is None

    def test_dial_pattern(self):
        target, _ = detect_call_intent("Dial 555-0100")
        assert target == "555-0100"

    def test_phone_pattern(self):
        target, _ = detect_call_intent("Phone Alice")
        assert target == "Alice"

    def test_no_match_returns_none(self):
        target, msg = detect_call_intent("What is the weather?")
        assert target is None
        assert msg is None

    def test_call_for_me(self):
        target, _ = detect_call_intent("Please call Bob for me")
        assert target == "Bob"


# ---------------------------------------------------------------------------
# is_phone_number
# ---------------------------------------------------------------------------


class TestIsPhoneNumber:
    def test_e164_number(self):
        assert is_phone_number("+15551234567") is True

    def test_plain_digits(self):
        assert is_phone_number("5551234567") is True

    def test_formatted_number(self):
        assert is_phone_number("(555) 123-4567") is True

    def test_name_is_not_phone(self):
        assert is_phone_number("John Smith") is False

    def test_short_string_not_phone(self):
        assert is_phone_number("123") is False


# ---------------------------------------------------------------------------
# lookup_contact — JSON
# ---------------------------------------------------------------------------


class TestLookupContactJson:
    def _make_contacts_file(self, contacts: list[dict]) -> str:
        fh = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump({"contacts": contacts}, fh)
        fh.close()
        return fh.name

    def test_exact_name_match(self):
        path = self._make_contacts_file([{"name": "Alice", "phone": "+15550001111"}])
        try:
            result = lookup_contact("Alice", path)
            assert result == "+15550001111"
        finally:
            os.unlink(path)

    def test_partial_name_match(self):
        path = self._make_contacts_file([{"name": "Alice Smith", "phone": "+15550001111"}])
        try:
            result = lookup_contact("Alice", path)
            assert result == "+15550001111"
        finally:
            os.unlink(path)

    def test_name_not_found_returns_none(self):
        path = self._make_contacts_file([{"name": "Bob", "phone": "+15550002222"}])
        try:
            result = lookup_contact("Alice", path)
            assert result is None
        finally:
            os.unlink(path)

    def test_missing_file_returns_none(self):
        result = lookup_contact("Alice", "/nonexistent/contacts.json")
        assert result is None

    def test_empty_contacts_file_path_returns_none(self):
        result = lookup_contact("Alice", "")
        assert result is None

    def test_case_insensitive(self):
        path = self._make_contacts_file([{"name": "alice", "phone": "+15550001111"}])
        try:
            result = lookup_contact("ALICE", path)
            assert result == "+15550001111"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# lookup_contact — vCard
# ---------------------------------------------------------------------------


class TestLookupContactVcf:
    _VCARD = (
        "BEGIN:VCARD\r\n"
        "VERSION:3.0\r\n"
        "FN:Bob Jones\r\n"
        "TEL;TYPE=CELL:+15559876543\r\n"
        "END:VCARD\r\n"
    )

    def _make_vcf(self, content: str) -> str:
        fh = tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False, encoding="utf-8")
        fh.write(content)
        fh.close()
        return fh.name

    def test_vcf_match(self):
        path = self._make_vcf(self._VCARD)
        try:
            result = lookup_contact("Bob", path)
            assert result == "+15559876543"
        finally:
            os.unlink(path)

    def test_vcf_no_match(self):
        path = self._make_vcf(self._VCARD)
        try:
            result = lookup_contact("Alice", path)
            assert result is None
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# log_call
# ---------------------------------------------------------------------------


class TestLogCall:
    def test_log_creates_file(self, tmp_path, monkeypatch):
        log_file = str(tmp_path / "outbound_calls.log")
        monkeypatch.setattr("rex.telephony.outbound._OUTBOUND_LOG_FILE", log_file)
        log_call("+15550001111", "initiated:CA123")
        assert os.path.isfile(log_file)

    def test_log_contains_number_and_outcome(self, tmp_path, monkeypatch):
        log_file = str(tmp_path / "outbound_calls.log")
        monkeypatch.setattr("rex.telephony.outbound._OUTBOUND_LOG_FILE", log_file)
        log_call("+15550001111", "initiated:CA123", message="Hello")
        with open(log_file, encoding="utf-8") as fh:
            record = json.loads(fh.readline())
        assert record["to"] == "+15550001111"
        assert record["outcome"] == "initiated:CA123"
        assert record["message"] == "Hello"
        assert "timestamp" in record

    def test_log_appends_multiple(self, tmp_path, monkeypatch):
        log_file = str(tmp_path / "outbound_calls.log")
        monkeypatch.setattr("rex.telephony.outbound._OUTBOUND_LOG_FILE", log_file)
        log_call("+1111", "ok")
        log_call("+2222", "failed")
        with open(log_file, encoding="utf-8") as fh:
            lines = fh.readlines()
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# make_call
# ---------------------------------------------------------------------------


class TestMakeCall:
    def _set_env(self, monkeypatch):
        monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACtest123")
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "authtest")
        monkeypatch.setenv("TWILIO_PHONE_NUMBER", "+10000000000")

    def test_requires_confirmation(self):
        result = make_call("+15550001111", confirmed=False)
        assert result["ok"] is False
        assert result["needs_confirmation"] is True
        assert "+15550001111" in result["prompt"]

    def test_no_credentials_returns_error(self, monkeypatch):
        monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
        monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("TWILIO_PHONE_NUMBER", raising=False)
        result = make_call("+15550001111", confirmed=True)
        assert result["ok"] is False
        assert "not configured" in result["error"]

    def test_successful_call_with_message(self, monkeypatch, tmp_path):
        self._set_env(monkeypatch)
        monkeypatch.setattr(
            "rex.telephony.outbound._OUTBOUND_LOG_FILE",
            str(tmp_path / "calls.log"),
        )

        mc, mc_cls, tw_mod, tw_rest_mod = _make_twilio_mock()
        mc.calls.create.return_value.sid = "CA_test_sid"

        with _patch_twilio_modules(mc, mc_cls, tw_mod, tw_rest_mod):
            result = make_call("+15550001111", message="Hello!", confirmed=True)

        assert result["ok"] is True
        assert result["call_sid"] == "CA_test_sid"
        assert result["to"] == "+15550001111"
        mc.calls.create.assert_called_once()

    def test_successful_call_without_message(self, monkeypatch, tmp_path):
        self._set_env(monkeypatch)
        monkeypatch.setattr(
            "rex.telephony.outbound._OUTBOUND_LOG_FILE",
            str(tmp_path / "calls.log"),
        )

        mc, mc_cls, tw_mod, tw_rest_mod = _make_twilio_mock()
        mc.calls.create.return_value.sid = "CA_no_msg"

        with _patch_twilio_modules(mc, mc_cls, tw_mod, tw_rest_mod):
            result = make_call("+15550001111", confirmed=True)

        assert result["ok"] is True
        _, kwargs = mc.calls.create.call_args
        assert "url" in kwargs

    def test_twilio_api_error_logged(self, monkeypatch, tmp_path):
        self._set_env(monkeypatch)
        log_file = str(tmp_path / "calls.log")
        monkeypatch.setattr("rex.telephony.outbound._OUTBOUND_LOG_FILE", log_file)

        mc, mc_cls, tw_mod, tw_rest_mod = _make_twilio_mock()
        mc.calls.create.side_effect = RuntimeError("API down")

        with _patch_twilio_modules(mc, mc_cls, tw_mod, tw_rest_mod):
            result = make_call("+15550001111", confirmed=True)

        assert result["ok"] is False
        assert "API down" in result["error"]

        with open(log_file, encoding="utf-8") as fh:
            record = json.loads(fh.readline())
        assert "error" in record["outcome"]

    def test_call_logged_on_success(self, monkeypatch, tmp_path):
        self._set_env(monkeypatch)
        log_file = str(tmp_path / "calls.log")
        monkeypatch.setattr("rex.telephony.outbound._OUTBOUND_LOG_FILE", log_file)

        mc, mc_cls, tw_mod, tw_rest_mod = _make_twilio_mock()
        mc.calls.create.return_value.sid = "CA_logged"

        with _patch_twilio_modules(mc, mc_cls, tw_mod, tw_rest_mod):
            make_call("+15550001111", message="Hi", confirmed=True)

        with open(log_file, encoding="utf-8") as fh:
            record = json.loads(fh.readline())
        assert record["to"] == "+15550001111"
        assert "initiated" in record["outcome"]
        assert record["message"] == "Hi"
