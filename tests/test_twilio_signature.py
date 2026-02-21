"""Tests for Twilio request signature verification (stdlib only).

All tests are offline and deterministic — no real network calls.
"""

from __future__ import annotations

from rex.messaging_backends.twilio_signature import (
    compute_twilio_signature,
    validate_twilio_signature,
)

# -- Known-good values (computed manually from the Twilio algorithm) --

_AUTH_TOKEN = "12345"
_URL = "https://mycompany.com/myapp.php?foo=1&bar=2"
_PARAMS = {
    "CallSid": "CA1234567890ABCDE",
    "Caller": "+14158675310",
    "Digits": "1234",
    "From": "+14158675310",
    "To": "+18005551212",
}


class TestComputeSignature:
    """Tests for compute_twilio_signature."""

    def test_deterministic_same_inputs(self) -> None:
        """Same inputs always produce the same signature."""
        sig1 = compute_twilio_signature(_AUTH_TOKEN, _URL, _PARAMS)
        sig2 = compute_twilio_signature(_AUTH_TOKEN, _URL, _PARAMS)
        assert sig1 == sig2
        assert isinstance(sig1, str)
        assert len(sig1) > 0

    def test_different_token_different_sig(self) -> None:
        """Changing the auth token changes the signature."""
        sig1 = compute_twilio_signature(_AUTH_TOKEN, _URL, _PARAMS)
        sig2 = compute_twilio_signature("other_token", _URL, _PARAMS)
        assert sig1 != sig2

    def test_different_url_different_sig(self) -> None:
        """Changing the URL changes the signature."""
        sig1 = compute_twilio_signature(_AUTH_TOKEN, _URL, _PARAMS)
        sig2 = compute_twilio_signature(_AUTH_TOKEN, "https://evil.example.com/hook", _PARAMS)
        assert sig1 != sig2

    def test_different_params_different_sig(self) -> None:
        """Changing the params changes the signature."""
        sig1 = compute_twilio_signature(_AUTH_TOKEN, _URL, _PARAMS)
        sig2 = compute_twilio_signature(_AUTH_TOKEN, _URL, {**_PARAMS, "Digits": "9999"})
        assert sig1 != sig2

    def test_empty_params(self) -> None:
        """Empty params dict produces a valid signature (just the URL)."""
        sig = compute_twilio_signature(_AUTH_TOKEN, _URL, {})
        assert isinstance(sig, str)
        assert len(sig) > 0

    def test_params_sorted_by_key(self) -> None:
        """Parameter order does not matter — they are sorted internally."""
        params_a = {"Z": "last", "A": "first"}
        params_b = {"A": "first", "Z": "last"}
        sig_a = compute_twilio_signature(_AUTH_TOKEN, _URL, params_a)
        sig_b = compute_twilio_signature(_AUTH_TOKEN, _URL, params_b)
        assert sig_a == sig_b


class TestValidateSignature:
    """Tests for validate_twilio_signature."""

    def test_valid_signature_accepted(self) -> None:
        """A correctly computed signature passes validation."""
        sig = compute_twilio_signature(_AUTH_TOKEN, _URL, _PARAMS)
        assert validate_twilio_signature(_AUTH_TOKEN, _URL, _PARAMS, sig) is True

    def test_wrong_signature_rejected(self) -> None:
        """A forged signature is rejected."""
        assert validate_twilio_signature(_AUTH_TOKEN, _URL, _PARAMS, "badbase64==") is False

    def test_wrong_token_rejected(self) -> None:
        """Using a different auth token rejects the signature."""
        sig = compute_twilio_signature(_AUTH_TOKEN, _URL, _PARAMS)
        assert validate_twilio_signature("wrong_token", _URL, _PARAMS, sig) is False

    def test_empty_auth_token_rejected(self) -> None:
        """An empty auth token always fails."""
        sig = compute_twilio_signature(_AUTH_TOKEN, _URL, _PARAMS)
        assert validate_twilio_signature("", _URL, _PARAMS, sig) is False

    def test_empty_signature_rejected(self) -> None:
        """An empty signature string always fails."""
        assert validate_twilio_signature(_AUTH_TOKEN, _URL, _PARAMS, "") is False

    def test_tampered_url_rejected(self) -> None:
        """Signature computed for one URL fails when validated against another."""
        sig = compute_twilio_signature(_AUTH_TOKEN, _URL, _PARAMS)
        tampered_url = "https://attacker.example.com/hook"
        assert validate_twilio_signature(_AUTH_TOKEN, tampered_url, _PARAMS, sig) is False

    def test_tampered_params_rejected(self) -> None:
        """Signature computed for one param set fails when params differ."""
        sig = compute_twilio_signature(_AUTH_TOKEN, _URL, _PARAMS)
        tampered = {**_PARAMS, "Body": "injected"}
        assert validate_twilio_signature(_AUTH_TOKEN, _URL, tampered, sig) is False


class TestSignatureBypassPrevention:
    """Ensure that no localhost/private-host bypass is possible."""

    def test_localhost_url_not_special(self) -> None:
        """A localhost URL does not get any special treatment."""
        url = "http://127.0.0.1/webhooks/twilio/sms"
        params = {"Body": "test", "From": "+1555", "To": "+1666"}
        sig = compute_twilio_signature(_AUTH_TOKEN, url, params)
        # Valid sig for localhost passes
        assert validate_twilio_signature(_AUTH_TOKEN, url, params, sig) is True
        # Wrong sig for localhost fails
        assert validate_twilio_signature(_AUTH_TOKEN, url, params, "bad") is False

    def test_private_ip_url_not_special(self) -> None:
        """A private IP URL does not bypass verification."""
        url = "http://10.0.0.1/webhooks/twilio/sms"
        params = {"Body": "test"}
        sig = compute_twilio_signature(_AUTH_TOKEN, url, params)
        assert validate_twilio_signature(_AUTH_TOKEN, url, params, sig) is True
        assert validate_twilio_signature(_AUTH_TOKEN, url, params, "bad") is False

    def test_ipv6_localhost_not_special(self) -> None:
        """IPv6 localhost does not bypass verification."""
        url = "http://[::1]/webhooks/twilio/sms"
        params = {"Body": "test"}
        sig = compute_twilio_signature(_AUTH_TOKEN, url, params)
        assert validate_twilio_signature(_AUTH_TOKEN, url, params, sig) is True
        assert validate_twilio_signature(_AUTH_TOKEN, url, params, "bad") is False
