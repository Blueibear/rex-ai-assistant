"""Device proof-of-possession crypto for mobile pairing (S5).

A pairing enrollment proves the mobile client holds the private half of a
P-256 (``secp256r1``) key pair by signing a *canonical transcript* that binds
the desktop identity, the specific challenge, the mobile public key, the
requested user, the requested scopes, and the one-time code.  The desktop
reconstructs the identical transcript from server-held state plus the
submitted public key/code and verifies the ECDSA signature.

Everything here is deterministic and side-effect free so the same inputs
always produce the same transcript bytes — enabling stable cross-repo
contract vectors with no wall-clock or network dependency.

Only the public key is ever parsed/stored server-side; the private key never
leaves the mobile device.  All failures raise :class:`ProofError` with a
stable, secret-free message.
"""

from __future__ import annotations

import base64
import hashlib
import json

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import Prehashed

# Domain-separation tag: prevents a signature produced for any other AskRex
# protocol from ever validating as a pairing proof.  Bump the version suffix
# if the transcript shape changes.
PROOF_DOMAIN = b"AskRex-Pairing-Proof-v1"
TRANSCRIPT_TYPE = "askrex-pairing-proof"
TRANSCRIPT_VERSION = 1

# Bounded input sizes for untrusted encodings (defense against oversized
# payloads before any crypto work).  A P-256 SPKI DER is ~120 bytes (~160 b64
# chars); a DER ECDSA signature is <= ~72 bytes (~100 b64 chars).
_MAX_PUBLIC_KEY_B64 = 512
_MAX_SIGNATURE_B64 = 256
_MAX_NONCE_B64 = 128


class ProofError(Exception):
    """A device proof, key, or signature is malformed or does not verify.

    The message is always a fixed, secret-free string safe to log or return.
    """


def _strict_b64decode(value: object, *, max_length: int, label: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ProofError(f"{label} is required.")
    if len(value) > max_length:
        raise ProofError(f"{label} is too large.")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ProofError(f"{label} is not valid base64.") from exc


def load_p256_public_key(public_key_b64: str) -> ec.EllipticCurvePublicKey:
    """Parse and strictly validate a base64 SPKI DER P-256 public key.

    Raises:
        ProofError: If the value is not strict base64, not a DER SubjectPublicKeyInfo,
            not an elliptic-curve key, or not on the ``secp256r1`` curve.
    """
    der = _strict_b64decode(public_key_b64, max_length=_MAX_PUBLIC_KEY_B64, label="Public key")
    try:
        key = serialization.load_der_public_key(der)
    except (ValueError, TypeError) as exc:
        raise ProofError("Public key is not a valid SPKI DER key.") from exc
    if not isinstance(key, ec.EllipticCurvePublicKey):
        raise ProofError("Public key is not an elliptic-curve key.")
    if not isinstance(key.curve, ec.SECP256R1):
        raise ProofError("Public key is not on the required P-256 curve.")
    return key


def canonical_public_key_spki_b64(key: ec.EllipticCurvePublicKey) -> str:
    """Return the canonical base64 SPKI DER encoding of *key*.

    Re-serializing removes any submitted-encoding ambiguity so the stored
    public key and its thumbprint are always canonical.
    """
    der = key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return base64.b64encode(der).decode("ascii")


def public_key_thumbprint(public_key_b64: str) -> str:
    """Return the SHA-256 hex thumbprint of a validated P-256 public key.

    The thumbprint is taken over the *canonical* SPKI DER (re-serialized from
    the parsed key), so two encodings of the same key always thumbprint alike.
    """
    key = load_p256_public_key(public_key_b64)
    canonical = canonical_public_key_spki_b64(key)
    der = base64.b64decode(canonical, validate=True)
    return hashlib.sha256(der).hexdigest()


def canonical_transcript(
    *,
    desktop_id: str,
    challenge_id: str,
    nonce_b64: str,
    mobile_public_key_b64: str,
    user_id: str,
    scopes: tuple[str, ...] | list[str],
    code: str,
) -> bytes:
    """Return the deterministic transcript bytes signed by the mobile client.

    The transcript is a domain-separated, canonically encoded JSON object
    (sorted keys, no insignificant whitespace, UTF-8).  ``scopes`` is sorted
    for a stable form.  Callers pass the *canonical* mobile public key so the
    same key always yields the same transcript.
    """
    payload = {
        "typ": TRANSCRIPT_TYPE,
        "v": TRANSCRIPT_VERSION,
        "desktop_id": desktop_id,
        "challenge_id": challenge_id,
        "nonce": nonce_b64,
        "mobile_public_key": mobile_public_key_b64,
        "user_id": user_id,
        "scopes": sorted(scopes),
        "code": code,
    }
    body = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return PROOF_DOMAIN + b"\n" + body


def verify_proof(
    *,
    public_key_b64: str,
    signature_b64: str,
    transcript: bytes,
) -> None:
    """Verify an ECDSA(P-256, SHA-256) signature over *transcript*.

    Raises:
        ProofError: If the key is invalid, the signature is not valid base64
            DER, or the signature does not verify against the transcript.
    """
    key = load_p256_public_key(public_key_b64)
    signature = _strict_b64decode(signature_b64, max_length=_MAX_SIGNATURE_B64, label="Signature")
    digest = hashlib.sha256(transcript).digest()
    try:
        key.verify(signature, digest, ec.ECDSA(Prehashed(hashes.SHA256())))
    except InvalidSignature as exc:
        raise ProofError("Proof signature does not verify.") from exc
    except (ValueError, TypeError) as exc:
        # Malformed DER signature bytes reach here.
        raise ProofError("Proof signature is malformed.") from exc


def decode_nonce(nonce_b64: str) -> bytes:
    """Strictly decode a base64 challenge nonce, bounding its size."""
    return _strict_b64decode(nonce_b64, max_length=_MAX_NONCE_B64, label="Nonce")


# ---------------------------------------------------------------------------
# Deterministic signing helpers (used by tests and contract-vector generation).
# These operate only on caller-supplied private keys; production server code
# never generates or holds a mobile private key.
# ---------------------------------------------------------------------------


def generate_p256_private_key() -> ec.EllipticCurvePrivateKey:
    """Generate a fresh P-256 private key (test/vector helper)."""
    return ec.generate_private_key(ec.SECP256R1())


def private_key_to_pkcs8_pem(private_key: ec.EllipticCurvePrivateKey) -> str:
    """Serialize a P-256 private key to unencrypted PKCS#8 PEM (test/vector helper)."""
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return str(pem.decode("ascii"))


def load_pkcs8_private_key(pem: str) -> ec.EllipticCurvePrivateKey:
    """Load a P-256 private key from unencrypted PKCS#8 PEM (test/vector helper)."""
    key = serialization.load_pem_private_key(pem.encode("ascii"), password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey) or not isinstance(key.curve, ec.SECP256R1):
        raise ProofError("Private key is not a valid P-256 key.")
    return key


def public_key_spki_b64(private_key: ec.EllipticCurvePrivateKey) -> str:
    """Return the canonical base64 SPKI DER public key for a private key."""
    return canonical_public_key_spki_b64(private_key.public_key())


def sign_transcript(private_key: ec.EllipticCurvePrivateKey, transcript: bytes) -> str:
    """Return the base64 DER ECDSA(P-256, SHA-256) signature over *transcript*."""
    digest = hashlib.sha256(transcript).digest()
    signature = private_key.sign(digest, ec.ECDSA(Prehashed(hashes.SHA256())))
    return base64.b64encode(signature).decode("ascii")


__all__ = [
    "PROOF_DOMAIN",
    "TRANSCRIPT_TYPE",
    "TRANSCRIPT_VERSION",
    "ProofError",
    "canonical_public_key_spki_b64",
    "canonical_transcript",
    "decode_nonce",
    "generate_p256_private_key",
    "load_p256_public_key",
    "load_pkcs8_private_key",
    "private_key_to_pkcs8_pem",
    "public_key_spki_b64",
    "public_key_thumbprint",
    "sign_transcript",
    "verify_proof",
]
