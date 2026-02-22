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
from collections.abc import Iterable, Mapping


def _normalize_params(
    params: Mapping[str, object] | Iterable[tuple[str, object]],
) -> list[tuple[str, str]]:
    """Normalize request parameters to a sorted list of key/value pairs.

    Twilio signs all form fields by appending key + value for each field,
    sorted alphabetically by key. Duplicate keys are valid in URL-encoded
    forms; each key/value pair must be included in the signature payload.
    """
    items: list[tuple[str, str]] = []

    if isinstance(params, Mapping):
        for key, value in params.items():
            if isinstance(value, (list, tuple)):
                for v in value:
                    items.append((str(key), "" if v is None else str(v)))
            else:
                items.append((str(key), "" if value is None else str(value)))
    else:
        for key, value in params:
            items.append((str(key), "" if value is None else str(value)))

    items.sort(key=lambda kv: kv[0])
    return items


def compute_twilio_signature(
    auth_token: str,
    url: str,
    params: Mapping[str, object] | Iterable[tuple[str, object]],
) -> str:
    """Compute the expected Twilio signature for a request.

    Args:
        auth_token: The Twilio Auth Token (secret).
        url: The full request URL (scheme + host + path).
        params: POST body parameters (key-value pairs).

    Returns:
        The Base64-encoded HMAC-SHA1 signature string.
    """
    # Sort params alphabetically by key and append to URL.
    data = url
    for key, value in _normalize_params(params):
        data += key + value

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
    params: Mapping[str, object] | Iterable[tuple[str, object]],
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
