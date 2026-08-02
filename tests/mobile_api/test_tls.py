"""Desktop TLS material tests (S7)."""

from __future__ import annotations

import ssl
from pathlib import Path

import pytest


class TestHostIsLoopback:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
    def test_loopback_hosts(self, host: str) -> None:
        from rex.mobile_api.tls import host_is_loopback

        assert host_is_loopback(host) is True

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.50", "not-an-ip"])  # nosec B104
    def test_non_loopback_hosts(self, host: str) -> None:
        from rex.mobile_api.tls import host_is_loopback

        assert host_is_loopback(host) is False


class TestEnsureServerTlsMaterial:
    def test_generates_loadable_material(self, tmp_path: Path) -> None:
        from rex.mobile_api.tls import ensure_server_tls_material

        material = ensure_server_tls_material(tmp_path)
        assert material.cert_path.exists()
        assert material.key_path.exists()
        assert len(material.fingerprint_sha256) == 64
        assert len(material.spki_pin_sha256_b64) == 44
        int(material.fingerprint_sha256, 16)  # valid hex
        context = material.build_ssl_context()
        assert isinstance(context, ssl.SSLContext)

    def test_idempotent_reuses_material_and_fingerprint(self, tmp_path: Path) -> None:
        from rex.mobile_api.tls import ensure_server_tls_material

        first = ensure_server_tls_material(tmp_path)
        first_cert_bytes = first.cert_path.read_bytes()
        second = ensure_server_tls_material(tmp_path)
        assert second.fingerprint_sha256 == first.fingerprint_sha256
        assert second.cert_path.read_bytes() == first_cert_bytes

    def test_different_data_dirs_get_different_fingerprints(self, tmp_path: Path) -> None:
        from rex.mobile_api.tls import ensure_server_tls_material

        one = ensure_server_tls_material(tmp_path / "a")
        two = ensure_server_tls_material(tmp_path / "b")
        assert one.fingerprint_sha256 != two.fingerprint_sha256

    def test_corrupted_key_fails_closed(self, tmp_path: Path) -> None:
        from rex.mobile_api.tls import MobileTlsConfigurationError, ensure_server_tls_material

        material = ensure_server_tls_material(tmp_path)
        material.key_path.write_bytes(b"not a real key")
        with pytest.raises(MobileTlsConfigurationError):
            ensure_server_tls_material(tmp_path)

    def test_missing_cert_with_present_key_fails_closed(self, tmp_path: Path) -> None:
        from rex.mobile_api.tls import MobileTlsConfigurationError, ensure_server_tls_material

        first = ensure_server_tls_material(tmp_path)
        first.cert_path.unlink()
        with pytest.raises(MobileTlsConfigurationError, match="incomplete"):
            ensure_server_tls_material(tmp_path)


class TestResolveMobileTls:
    def test_loopback_without_require_tls_returns_none_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        from rex.mobile_api.tls import resolve_mobile_tls

        material = resolve_mobile_tls(host="127.0.0.1", require_tls=False, data_dir=tmp_path)
        assert material is None
        assert not (tmp_path / "mobile_tls").exists()

    def test_loopback_with_require_tls_provisions_material(self, tmp_path: Path) -> None:
        from rex.mobile_api.tls import resolve_mobile_tls

        material = resolve_mobile_tls(host="localhost", require_tls=True, data_dir=tmp_path)
        assert material is not None
        assert material.cert_path.exists()

    def test_non_loopback_always_provisions_regardless_of_require_tls_flag(
        self, tmp_path: Path
    ) -> None:
        from rex.mobile_api.tls import resolve_mobile_tls

        material = resolve_mobile_tls(
            host="0.0.0.0",
            require_tls=False,
            data_dir=tmp_path,
            advertised_host="192.168.1.50",  # nosec B104
        )
        assert material is not None
        assert material.cert_path.exists()

    def test_non_loopback_fails_closed_when_material_cannot_be_provisioned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import rex.mobile_api.tls as tls_module
        from rex.mobile_api.tls import MobileTlsConfigurationError

        def _boom(*_args, **_kwargs):
            raise MobileTlsConfigurationError("simulated provisioning failure")

        monkeypatch.setattr(tls_module, "ensure_server_tls_material", _boom)
        with pytest.raises(MobileTlsConfigurationError):
            tls_module.resolve_mobile_tls(host="192.168.1.50", require_tls=False, data_dir=tmp_path)


def test_wildcard_bind_requires_advertised_host(tmp_path: Path) -> None:
    from rex.mobile_api.tls import MobileTlsConfigurationError, resolve_mobile_tls

    with pytest.raises(MobileTlsConfigurationError, match="concrete advertised"):
        resolve_mobile_tls(host="0.0.0.0", require_tls=False, data_dir=tmp_path)  # nosec B104


def test_binding_contains_exact_endpoint_fingerprint_and_spki(tmp_path: Path) -> None:
    from rex.mobile_api.tls import ensure_server_tls_material

    material = ensure_server_tls_material(tmp_path, certificate_host="192.168.1.50")
    binding = material.binding(advertised_host="192.168.1.50", advertised_port=8765)
    assert binding.server_url == "https://192.168.1.50:8765"
    assert binding.certificate_fingerprint == material.fingerprint_sha256
    assert binding.spki_pins == (material.spki_pin_sha256_b64,)
