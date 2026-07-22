from rex.response.builder import home_assistant_status_message


def test_verified_language_is_confident() -> None:
    assert home_assistant_status_message("verified", entity_id="light.kitchen").startswith(
        "Confirmed"
    )


def test_unverified_language_discloses_uncertainty() -> None:
    message = home_assistant_status_message("attempted_unverified", entity_id="light.kitchen")
    assert "tried" in message
    assert "could not verify" in message


def test_denied_and_failed_language_do_not_claim_completion() -> None:
    denied = home_assistant_status_message(
        "denied", entity_id="lock.front_door", detail="Confirmation expired."
    )
    failed = home_assistant_status_message(
        "failed", entity_id="switch.garage", detail="Connection timed out."
    )
    assert "did not change" in denied
    assert "failed" in failed
    assert "Confirmed" not in denied + failed
