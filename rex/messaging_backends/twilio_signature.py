"""Twilio request signature verification using stdlib only.

Implements the Twilio signature validation algorithm so that the inbound
webhook can confirm that incoming requests originate from Twilio.

Algorithm (from Twilio docs):
1. Take the full URL of the request (including scheme, host, port, path).
2. If the request is a POST with ``application/x-www-form-urlencoded`` body,
   sort the POST parameters alphabetically by key and append each key-value
   pair to the URL (no separator, no encoding).
3. Compute an HMAC-SHA1 of the resulting string using the Twilio Auth Token
   as the key.
4. Base64-encode the HMAC digest and compare to the ``X-Twilio-Signature``
   header value.

No new dependencies are introduced — this uses only ``hmac``, ``hashlib``,
and ``base64`` from the standard library.
"""

from __future__ import annotations

import base64
import hashlib
import hmac


def compute_twilio_signature(auth_token: str, url: str, params: dict[str, str]) -> str:
    """Compute the expected Twilio signature for a request.

    Args:
        auth_token: The Twilio Auth Token (secret).
        url: The full request URL (scheme + host + path).
        params: POST body parameters (key-value pairs).

    Returns:
        The Base64-encoded HMAC-SHA1 signature string.
    """
    # Sort params alphabetically by key and append to URL
    data = url
    for key in sorted(params.keys()):
        data += key + params[key]

    # Compute HMAC-SHA1
    mac = hmac.new(
        auth_token.encode("utf-8"),
        data.encode("utf-8"),
        hashlib.sha1,
    )
    return base64.b64encode(mac.digest()).decode("utf-8")


def validate_twilio_signature(
    auth_token: str,
    url: str,
    params: dict[str, str],
    signature: str,
) -> bool:
    """Validate a Twilio request signature.

    Uses constant-time comparison to prevent timing attacks.

    Args:
        auth_token: The Twilio Auth Token (secret).
        url: The full request URL (scheme + host + path).
        params: POST body parameters (key-value pairs).
        signature: The ``X-Twilio-Signature`` header value.

    Returns:
        ``True`` if the signature is valid, ``False`` otherwise.
    """
    if not auth_token or not signature:
        return False

    expected = compute_twilio_signature(auth_token, url, params)
    return hmac.compare_digest(expected, signature)


__all__ = [
    "compute_twilio_signature",
    "validate_twilio_signature",
]
