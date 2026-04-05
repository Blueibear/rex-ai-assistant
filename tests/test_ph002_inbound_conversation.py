"""Tests for US-PH-002: Inbound call STT + LLM conversation loop."""

from __future__ import annotations

from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def twilio_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set fake Twilio credentials in the environment."""
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACfake_account_sid")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "fake_auth_token_1234")
    monkeypatch.setenv("TWILIO_PHONE_NUMBER", "+15550001234")


@pytest.fixture()
def app_client(twilio_env):
    """Flask test client with telephony blueprint registered."""
    from flask import Flask

    from rex.telephony.twilio_handler import create_blueprint

    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.register_blueprint(create_blueprint())
    return flask_app.test_client()


# ---------------------------------------------------------------------------
# _detect_voicemail_intent
# ---------------------------------------------------------------------------


def test_detect_voicemail_intent_true_for_leave_a_message() -> None:
    from rex.telephony.twilio_handler import _detect_voicemail_intent

    assert _detect_voicemail_intent("I'd like to leave a message")


def test_detect_voicemail_intent_true_for_voicemail() -> None:
    from rex.telephony.twilio_handler import _detect_voicemail_intent

    assert _detect_voicemail_intent("Can I leave voicemail?")


def test_detect_voicemail_intent_false_for_normal_speech() -> None:
    from rex.telephony.twilio_handler import _detect_voicemail_intent

    assert not _detect_voicemail_intent("What is the weather today?")


# ---------------------------------------------------------------------------
# _detect_transfer_intent
# ---------------------------------------------------------------------------


def test_detect_transfer_intent_true() -> None:
    from rex.telephony.twilio_handler import _detect_transfer_intent

    assert _detect_transfer_intent("I want to speak to a human")


def test_detect_transfer_intent_false() -> None:
    from rex.telephony.twilio_handler import _detect_transfer_intent

    assert not _detect_transfer_intent("What time is it?")


# ---------------------------------------------------------------------------
# _save_voicemail
# ---------------------------------------------------------------------------


def test_save_voicemail_creates_file(tmp_path, monkeypatch) -> None:
    import rex.telephony.twilio_handler as th

    monkeypatch.setattr(th, "_VOICEMAIL_DIR", str(tmp_path / "voicemail"))
    path = th._save_voicemail("+15559990000", "Hello, this is a test message.")
    assert path.endswith(".txt")
    content = open(path).read()
    assert "Hello, this is a test message." in content
    assert "+15559990000" in content


def test_save_voicemail_filename_contains_caller_digits(tmp_path, monkeypatch) -> None:
    import rex.telephony.twilio_handler as th

    monkeypatch.setattr(th, "_VOICEMAIL_DIR", str(tmp_path / "voicemail"))
    path = th._save_voicemail("+15551234567", "test")
    import os

    assert "15551234567" in os.path.basename(path)


# ---------------------------------------------------------------------------
# _build_gather_twiml
# ---------------------------------------------------------------------------


def test_build_gather_twiml_contains_gather_element() -> None:
    from rex.telephony.twilio_handler import _build_gather_twiml

    twiml = _build_gather_twiml("Hello caller", "/telephony/inbound/gather?turn=1")
    assert "Gather" in twiml or "gather" in twiml.lower()


def test_build_gather_twiml_contains_say_text() -> None:
    from rex.telephony.twilio_handler import _build_gather_twiml

    twiml = _build_gather_twiml("Hello caller", "/telephony/inbound/gather?turn=1")
    assert "Hello caller" in twiml


# ---------------------------------------------------------------------------
# /telephony/inbound/call — now returns <Gather>
# ---------------------------------------------------------------------------


def test_inbound_call_returns_gather(app_client) -> None:
    with patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=True):
        resp = app_client.post(
            "/telephony/inbound/call",
            data={"CallSid": "CAtest", "From": "+15559990000"},
        )
    assert resp.status_code == 200
    assert b"Gather" in resp.data or b"gather" in resp.data.lower()


def test_inbound_call_action_points_to_gather_route(app_client) -> None:
    with patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=True):
        resp = app_client.post(
            "/telephony/inbound/call",
            data={"CallSid": "CAtest", "From": "+15559990000"},
        )
    assert b"/telephony/inbound/gather" in resp.data


# ---------------------------------------------------------------------------
# /telephony/inbound/gather — conversation loop
# ---------------------------------------------------------------------------


def test_gather_returns_xml(app_client) -> None:
    with (
        patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=True),
        patch("rex.telephony.twilio_handler._generate_reply", return_value="Sure!"),
    ):
        resp = app_client.post(
            "/telephony/inbound/gather?turn=1",
            data={"SpeechResult": "What time is it?", "From": "+15559990000"},
        )
    assert resp.status_code == 200
    assert "xml" in resp.content_type


def test_gather_calls_generate_reply(app_client) -> None:
    with (
        patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=True),
        patch(
            "rex.telephony.twilio_handler._generate_reply", return_value="It is noon."
        ) as mock_gen,
    ):
        app_client.post(
            "/telephony/inbound/gather?turn=1",
            data={"SpeechResult": "What time is it?", "From": "+15559990000"},
        )
    mock_gen.assert_called_once_with("What time is it?")


def test_gather_increments_turn_in_action(app_client) -> None:
    with (
        patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=True),
        patch("rex.telephony.twilio_handler._generate_reply", return_value="Here you go."),
    ):
        resp = app_client.post(
            "/telephony/inbound/gather?turn=2",
            data={"SpeechResult": "Tell me a joke", "From": "+15559990000"},
        )
    # Next <Gather> action should reference turn=3
    assert b"turn=3" in resp.data


def test_gather_ends_after_max_turns(app_client) -> None:
    with patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=True):
        resp = app_client.post(
            "/telephony/inbound/gather?turn=6",
            data={"SpeechResult": "still here", "From": "+15559990000"},
        )
    assert resp.status_code == 200
    # Should hang up gracefully — no further <Gather>
    assert b"Gather" not in resp.data or b"Goodbye" in resp.data


def test_gather_no_speech_prompts_again(app_client) -> None:
    with patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=True):
        resp = app_client.post(
            "/telephony/inbound/gather?turn=1",
            data={"SpeechResult": "", "From": "+15559990000"},
        )
    assert resp.status_code == 200
    assert b"Gather" in resp.data


def test_gather_voicemail_intent_redirects_to_record(app_client) -> None:
    with (patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=True),):
        resp = app_client.post(
            "/telephony/inbound/gather?turn=1",
            data={"SpeechResult": "I'd like to leave a message", "From": "+15559990000"},
        )
    assert resp.status_code == 200
    # Should contain <Record> verb
    assert b"Record" in resp.data or b"record" in resp.data.lower()


def test_gather_transfer_intent_with_transfer_number(app_client, monkeypatch) -> None:
    monkeypatch.setenv("TWILIO_TRANSFER_NUMBER", "+15550009999")
    with (patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=True),):
        resp = app_client.post(
            "/telephony/inbound/gather?turn=1",
            data={"SpeechResult": "transfer me to a human", "From": "+15559990000"},
        )
    assert resp.status_code == 200
    assert b"Dial" in resp.data or b"+15550009999" in resp.data


def test_gather_transfer_intent_without_transfer_number_falls_through(
    app_client, monkeypatch
) -> None:
    monkeypatch.delenv("TWILIO_TRANSFER_NUMBER", raising=False)
    with (
        patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=True),
        patch("rex.telephony.twilio_handler._generate_reply", return_value="I cannot transfer."),
    ):
        resp = app_client.post(
            "/telephony/inbound/gather?turn=1",
            data={"SpeechResult": "transfer me to a real person", "From": "+15559990000"},
        )
    assert resp.status_code == 200
    assert b"Dial" not in resp.data


# ---------------------------------------------------------------------------
# /telephony/inbound/voicemail — saves transcript
# ---------------------------------------------------------------------------


def test_voicemail_route_saves_transcript(app_client, tmp_path, monkeypatch) -> None:
    import rex.telephony.twilio_handler as th

    monkeypatch.setattr(th, "_VOICEMAIL_DIR", str(tmp_path / "voicemail"))
    with patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=True):
        resp = app_client.post(
            "/telephony/inbound/voicemail",
            data={
                "TranscriptionText": "Hello Rex, call me back.",
                "From": "+15559990000",
            },
        )
    assert resp.status_code == 200
    files = list((tmp_path / "voicemail").glob("*.txt"))
    assert len(files) == 1
    assert "Hello Rex, call me back." in files[0].read_text()


def test_voicemail_route_returns_xml(app_client, tmp_path, monkeypatch) -> None:
    import rex.telephony.twilio_handler as th

    monkeypatch.setattr(th, "_VOICEMAIL_DIR", str(tmp_path / "voicemail"))
    with patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=True):
        resp = app_client.post(
            "/telephony/inbound/voicemail",
            data={"TranscriptionText": "test", "From": "+15559990000"},
        )
    assert "xml" in resp.content_type
