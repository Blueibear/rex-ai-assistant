from __future__ import annotations

from unittest.mock import patch

import pytest

from rex.assistant import Assistant
from rex.media.accounts import MediaAccountStore
from rex.media.registry import AudioTargetRegistry
from rex.output_routing import OutputRoutingService, UserOutputPolicy
from rex.runtime.invocation import turn_invocation
from rex.runtime.turn import IdentityResolution, TurnSource
from rex.voice_identity.fallback_flow import resolve_speaker_identity
from rex.voice_identity.types import RecognitionDecision, RecognitionResult

_JAMES_CRED = "cred_" + ("a" * 32)
_COLE_CRED = "cred_" + ("b" * 32)


def _service(tmp_path) -> tuple[OutputRoutingService, MediaAccountStore]:
    accounts = MediaAccountStore(root=tmp_path / "accounts")
    accounts.put(
        "james",
        "apple_music",
        "james-main",
        _JAMES_CRED,
        "James Apple Music",
    )
    accounts.put(
        "cole",
        "spotify",
        "cole-main",
        _COLE_CRED,
        "Cole Spotify",
    )
    service = OutputRoutingService(
        AudioTargetRegistry(()),
        root=tmp_path / "routing",
        media_accounts=accounts,
        household_media_path=tmp_path / "household-media.json",
    )
    service.save_policy(
        "james",
        UserOutputPolicy(
            default_media_provider="apple_music",
            default_media_account_id="james-main",
        ),
    )
    service.save_policy(
        "cole",
        UserOutputPolicy(
            default_media_provider="spotify",
            default_media_account_id="cole-main",
        ),
    )
    service.set_household_primary_media_account(
        owner_user_id="james",
        provider="apple_music",
        account_id="james-main",
    )
    return service, accounts


def test_recognized_speaker_uses_only_their_default_account(tmp_path) -> None:
    service, _accounts = _service(tmp_path)

    james = service.resolve_media_account(
        active_user_id="james",
        identity_resolution=IdentityResolution.VOICE_RECOGNIZED,
        requested_account_id=None,
        operation="play",
    )
    cole = service.resolve_media_account(
        active_user_id="cole",
        identity_resolution=IdentityResolution.VOICE_RECOGNIZED,
        requested_account_id=None,
        operation="play",
    )

    assert james is not None
    assert james.user_id == "james"
    assert james.account_id == "james-main"
    assert cole is not None
    assert cole.user_id == "cole"
    assert cole.account_id == "cole-main"


def test_explicit_profile_identity_uses_own_account(tmp_path) -> None:
    service, _accounts = _service(tmp_path)

    account = service.resolve_media_account(
        active_user_id="cole",
        identity_resolution=IdentityResolution.EXPLICIT,
        requested_account_id=None,
        operation="favorite",
    )

    assert account is not None
    assert account.user_id == "cole"


def test_unresolved_speaker_may_use_household_primary_for_playback(tmp_path) -> None:
    service, _accounts = _service(tmp_path)

    account = service.resolve_media_account(
        active_user_id=None,
        identity_resolution=IdentityResolution.UNKNOWN,
        requested_account_id=None,
        operation="play",
    )

    assert account is not None
    assert account.user_id == "james"
    assert account.account_id == "james-main"


@pytest.mark.parametrize(
    "identity",
    [IdentityResolution.UNKNOWN, IdentityResolution.FALLBACK],
)
def test_unresolved_speaker_cannot_mutate_household_primary_library(
    tmp_path,
    identity: IdentityResolution,
) -> None:
    service, _accounts = _service(tmp_path)

    with pytest.raises(PermissionError, match="library or profile mutations"):
        service.resolve_media_account(
            active_user_id=("james" if identity is IdentityResolution.FALLBACK else None),
            identity_resolution=identity,
            requested_account_id=None,
            operation="favorite",
        )


def test_requested_private_account_requires_strong_identity(tmp_path) -> None:
    service, _accounts = _service(tmp_path)

    with pytest.raises(PermissionError, match="trusted user identity"):
        service.resolve_media_account(
            active_user_id="james",
            identity_resolution=IdentityResolution.FALLBACK,
            requested_account_id="james-main",
            operation="play",
        )


def test_explicit_account_cannot_select_another_users_private_account(tmp_path) -> None:
    service, _accounts = _service(tmp_path)

    with pytest.raises(PermissionError, match="not owned by the active user"):
        service.resolve_media_account(
            active_user_id="james",
            identity_resolution=IdentityResolution.EXPLICIT,
            requested_account_id="cole-main",
            operation="play",
        )


def test_household_policy_persists_only_non_secret_account_reference(tmp_path) -> None:
    _service(tmp_path)
    household_path = tmp_path / "household-media.json"

    text = household_path.read_text(encoding="utf-8")

    assert "james-main" in text
    assert "apple_music" in text
    assert _JAMES_CRED not in text
    assert "credential_ref" not in text


def _recognition(
    decision: RecognitionDecision,
    *,
    user_id: str | None,
    score: float,
) -> RecognitionResult:
    return RecognitionResult(
        decision=decision,
        best_user_id=user_id,
        score=score,
        accept_threshold=0.85,
        review_threshold=0.65,
    )


def test_recognized_voice_provenance_reaches_assistant_turn_context() -> None:
    result = _recognition(
        RecognitionDecision.RECOGNIZED,
        user_id="james",
        score=0.96,
    )
    with patch("rex.voice_identity.fallback_flow.set_session_user"):
        assert resolve_speaker_identity(result) == "james"

    assistant = Assistant.__new__(Assistant)
    with turn_invocation(TurnSource.VOICE):
        context = assistant._build_turn_context("james", voice_mode=True)

    assert context.identity_resolution is IdentityResolution.VOICE_RECOGNIZED


def test_review_match_provenance_reaches_turn_context() -> None:
    result = _recognition(
        RecognitionDecision.REVIEW,
        user_id="cole",
        score=0.72,
    )
    with patch(
        "rex.voice_identity.fallback_flow.resolve_active_user",
        return_value="cole",
    ):
        assert resolve_speaker_identity(result) == "cole"

    assistant = Assistant.__new__(Assistant)
    with turn_invocation(TurnSource.VOICE):
        context = assistant._build_turn_context("cole", voice_mode=True)

    assert context.identity_resolution is IdentityResolution.VOICE_REVIEW


def test_review_mismatch_is_fallback_not_voice_authority() -> None:
    result = _recognition(
        RecognitionDecision.REVIEW,
        user_id="cole",
        score=0.72,
    )
    with patch(
        "rex.voice_identity.fallback_flow.resolve_active_user",
        return_value="james",
    ):
        assert resolve_speaker_identity(result) == "james"

    assistant = Assistant.__new__(Assistant)
    with turn_invocation(TurnSource.VOICE):
        context = assistant._build_turn_context("james", voice_mode=True)

    assert context.identity_resolution is IdentityResolution.FALLBACK


def test_typed_surface_defaults_to_explicit_identity_provenance() -> None:
    assistant = Assistant.__new__(Assistant)
    with turn_invocation(TurnSource.ELECTRON):
        context = assistant._build_turn_context("james", voice_mode=False)

    assert context.identity_resolution is IdentityResolution.EXPLICIT
