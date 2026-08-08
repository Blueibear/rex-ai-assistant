from __future__ import annotations

from pathlib import Path

import pytest

from rex.response.builder import home_assistant_status_message


def test_verified_language_names_observed_state_confidently() -> None:
    message = home_assistant_status_message(
        "verified",
        entity_id="light.kitchen",
        expected={"state": "on", "attributes": {}},
    )
    assert message == "Confirmed light kitchen is on."


def test_attempted_unverified_language_discloses_uncertainty() -> None:
    message = home_assistant_status_message("attempted_unverified", entity_id="light.kitchen")
    assert message.startswith("I tried")
    assert "could not verify" in message


def test_completed_language_reports_dispatch_without_claiming_verification() -> None:
    message = home_assistant_status_message("completed", entity_id="scene.movie_night")
    assert message == "I asked HA to update scene movie night."
    assert "Confirmed" not in message


def test_failed_language_explains_failure_without_claiming_success() -> None:
    message = home_assistant_status_message(
        "failed", entity_id="switch.garage", detail="Connection timed out."
    )
    assert message == "That failed because Connection timed out."
    assert "Confirmed" not in message


@pytest.mark.parametrize(
    "status",
    ["attempted_unverified", "completed", "confirmation_required", "denied", "failed"],
)
def test_only_verified_status_uses_confirmed_language(status: str) -> None:
    message = home_assistant_status_message(
        status,
        entity_id="lock.front_door",
        detail="The operation did not produce verified state evidence.",
    )
    assert "Confirmed" not in message


def test_denied_language_states_that_no_change_was_made() -> None:
    message = home_assistant_status_message(
        "denied", entity_id="lock.front_door", detail="Confirmation expired."
    )
    assert message.startswith("I did not change")


def test_readme_documents_home_assistant_verification_vocabulary() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for phrase in ("I tried", "I asked HA to", "Confirmed", "That failed because"):
        assert phrase in readme
