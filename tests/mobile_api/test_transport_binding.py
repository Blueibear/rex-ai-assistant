"""S7: desktop TLS certificate binding enforced at S6 device-session activation.

A paired device's grant is bound to the desktop TLS certificate fingerprint
that was active at pairing-approval time. Loopback (no-TLS) activation is
unaffected — this only applies once the gateway is actually serving TLS.
"""

from __future__ import annotations

from pathlib import Path

from rex.mobile_api.tls import TransportBinding
from tests.mobile_api.conftest import auth_header, create_user, login_tokens

TEST_BINDING = TransportBinding(
    server_url="https://rex.example.test:8765",
    certificate_fingerprint="a" * 64,
    spki_pins=("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",),
)


def _build_app(mobile_env: Path, clock, *, binding=TEST_BINDING):
    from rex.config import MobileApiConfig
    from rex.mobile_api.app import create_mobile_app
    from rex.mobile_api.db import migrate_users_db
    from rex.mobile_api.pairing import PairingAuthority
    from rex.mobile_api.services import MobileApiServices

    db_path = mobile_env / "users.db"
    migrate_users_db(db_path)
    services = MobileApiServices.build(MobileApiConfig(), db_path=db_path, clock=clock)
    services.transport_binding = binding
    services.pairing_authority = PairingAuthority(
        db_path, clock=clock, transport_binding_provider=lambda: binding
    )
    app = create_mobile_app(services=services)
    app.config["TESTING"] = True
    return app, services


def _pair_device(client, services, username: str, password: str) -> dict:
    """Run the S5/S6 flow through device-challenge, stopping before activation."""
    from rex.mobile_api.db import connect
    from rex.mobile_api.device_proof import (
        canonical_session_transcript,
        canonical_transcript,
        generate_p256_private_key,
        public_key_spki_b64,
        sign_transcript,
    )

    create_user(username, password)
    bootstrap = login_tokens(client, username, password)
    conn = connect(services.db_path)
    try:
        user = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    finally:
        conn.close()
    user_id = str(user["id"])
    private_key = generate_p256_private_key()
    public_key = public_key_spki_b64(private_key)
    scopes = ["chat.send"]
    challenge = services.pairing_authority.create_challenge(user_id=user_id, scopes=scopes)
    transcript = canonical_transcript(
        desktop_id=challenge.desktop_id,
        challenge_id=challenge.challenge_id,
        nonce_b64=challenge.nonce_b64,
        mobile_public_key_b64=public_key,
        user_id=challenge.user_id,
        scopes=challenge.scopes,
        code=challenge.code,
        server_url=challenge.server_url,
        certificate_fingerprint=challenge.certificate_fingerprint,
        spki_pins=challenge.spki_pins,
    )
    submitted = services.pairing_authority.submit_proof(
        {
            "desktop_id": challenge.desktop_id,
            "challenge_id": challenge.challenge_id,
            "nonce": challenge.nonce_b64,
            "code": challenge.code,
            "user_id": challenge.user_id,
            "scopes": list(challenge.scopes),
            "public_key": public_key,
            "signature": sign_transcript(private_key, transcript),
            "device_name": "Test Device",
            "platform": "ios",
        }
    )
    grant = services.pairing_authority.approve(submitted.request_id, approved_by="TEST\\Owner")
    challenge_response = client.post(
        "/mobile/auth/device-challenge",
        json={"device_id": grant.device_id, "grant_id": grant.grant_id},
        headers=auth_header(bootstrap["access_token"]),
    )
    assert challenge_response.status_code == 200, challenge_response.get_json()
    activation = challenge_response.get_json()
    activation_transcript = canonical_session_transcript(
        desktop_id=activation["desktop_id"],
        bootstrap_session_id=activation["bootstrap_session_id"],
        challenge_id=activation["challenge_id"],
        nonce_b64=activation["nonce"],
        device_id=activation["device_id"],
        grant_id=activation["grant_id"],
        grant_version=activation["grant_version"],
        user_id=activation["user_id"],
    )
    return {
        "bootstrap": bootstrap,
        "challenge_id": activation["challenge_id"],
        "signature": sign_transcript(private_key, activation_transcript),
        "device_id": grant.device_id,
    }


def _activate(client, pending: dict):
    return client.post(
        "/mobile/auth/activate-device",
        json={"challenge_id": pending["challenge_id"], "signature": pending["signature"]},
        headers=auth_header(pending["bootstrap"]["access_token"]),
    )


class TestTransportBindingAtActivation:
    def test_loopback_activation_ignores_binding(self, mobile_env: Path, clock) -> None:
        app, services = _build_app(mobile_env, clock)
        with app.test_client() as client:
            pending = _pair_device(client, services, "loop-user", "correct horse battery staple")
            services.transport_binding = None
            response = _activate(client, pending)
        assert response.status_code == 200, response.get_json()

    def test_matching_fingerprint_activates(self, mobile_env: Path, clock) -> None:
        app, services = _build_app(mobile_env, clock)
        with app.test_client() as client:
            pending = _pair_device(client, services, "match-user", "correct horse battery staple")
        services.transport_binding = TEST_BINDING
        with app.test_client() as client:
            response = _activate(client, pending)
        assert response.status_code == 200, response.get_json()

    def test_mismatched_fingerprint_fails_closed(self, mobile_env: Path, clock) -> None:
        app, services = _build_app(mobile_env, clock)
        with app.test_client() as client:
            pending = _pair_device(
                client, services, "mismatch-user", "correct horse battery staple"
            )
        services.transport_binding = TransportBinding(
            TEST_BINDING.server_url, "f" * 64, TEST_BINDING.spki_pins
        )
        with app.test_client() as client:
            response = _activate(client, pending)
        assert response.status_code == 403
        assert response.get_json()["error"]["code"] == "PAIRING_INVALID"

    def test_unbound_legacy_device_fails_closed_under_tls(self, mobile_env: Path, clock) -> None:
        from rex.mobile_api.db import connect

        app, services = _build_app(mobile_env, clock)
        with app.test_client() as client:
            pending = _pair_device(client, services, "legacy-user", "correct horse battery staple")
        conn = connect(services.db_path)
        try:
            conn.execute(
                """UPDATE mobile_paired_devices
                   SET desktop_cert_fingerprint = '', server_url = '', spki_pins_json = '[]'
                   WHERE device_id = ?""",
                (pending["device_id"],),
            )
        finally:
            conn.close()
        services.transport_binding = TEST_BINDING
        with app.test_client() as client:
            response = _activate(client, pending)
        assert response.status_code == 403
        assert response.get_json()["error"]["code"] == "PAIRING_INVALID"
