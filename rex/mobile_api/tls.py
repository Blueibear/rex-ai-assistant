"""Fail-closed desktop TLS material and transport bindings for the mobile API."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import os
import ssl
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlunsplit

CERT_FILENAME = "server_cert.pem"
KEY_FILENAME = "server_key.pem"
LOCK_FILENAME = ".tls.lock"
TLS_SUBDIR = "mobile_tls"
CERT_VALIDITY_DAYS = 3650
_LOCK_TIMEOUT_SECONDS = 10.0
_LOOPBACK_HOSTS = {"localhost"}
_WILDCARD_HOSTS = {"0.0.0.0", "::", "*"}


class MobileTlsConfigurationError(RuntimeError):
    """TLS or its immutable client binding cannot be trusted."""


def host_is_loopback(host: str) -> bool:
    value = host.strip().strip("[]").lower()
    if value in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def host_is_wildcard(host: str) -> bool:
    return host.strip().strip("[]").lower() in _WILDCARD_HOSTS


def _canonical_host(host: str) -> str:
    value = host.strip().strip("[]").lower()
    if not value or host_is_wildcard(value):
        raise MobileTlsConfigurationError("A concrete advertised mobile host is required.")
    return value


def advertised_server_url(host: str, port: int) -> str:
    value = _canonical_host(host)
    if not 1 <= port <= 65535:
        raise MobileTlsConfigurationError("The advertised mobile port is invalid.")
    try:
        parsed = ipaddress.ip_address(value)
        netloc = f"[{value}]:{port}" if parsed.version == 6 else f"{value}:{port}"
    except ValueError:
        netloc = f"{value}:{port}"
    return urlunsplit(("https", netloc, "", "", ""))


@dataclass(frozen=True)
class TransportBinding:
    server_url: str
    certificate_fingerprint: str
    spki_pins: tuple[str, ...]


@dataclass(frozen=True)
class TlsMaterial:
    cert_path: Path
    key_path: Path
    certificate_host: str
    fingerprint_sha256: str
    spki_pin_sha256_b64: str

    def build_ssl_context(self) -> ssl.SSLContext:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        try:
            context.load_cert_chain(str(self.cert_path), str(self.key_path))
        except (ssl.SSLError, OSError) as exc:
            raise MobileTlsConfigurationError(
                "TLS certificate material could not be loaded."
            ) from exc
        return context

    def binding(self, *, advertised_host: str, advertised_port: int) -> TransportBinding:
        host = _canonical_host(advertised_host)
        if host != self.certificate_host:
            raise MobileTlsConfigurationError(
                "The advertised host does not match the TLS certificate."
            )
        return TransportBinding(
            server_url=advertised_server_url(host, advertised_port),
            certificate_fingerprint=self.fingerprint_sha256,
            spki_pins=(self.spki_pin_sha256_b64,),
        )


def _tls_dir(data_dir: Path) -> Path:
    return Path(data_dir) / TLS_SUBDIR


def _secure_private_key(path: Path) -> None:
    if os.name == "nt":
        try:
            from rex.credential_vault import _harden_file_acl  # noqa: PLC2701

            _harden_file_acl(path)
        except Exception as exc:
            raise MobileTlsConfigurationError(
                "The TLS private-key ACL could not be secured."
            ) from exc
        return
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise OSError("unexpected key mode")
    except OSError as exc:
        raise MobileTlsConfigurationError(
            "The TLS private-key permissions could not be secured."
        ) from exc


@contextmanager
def _generation_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / LOCK_FILENAME
    handle: BinaryIO = open(lock_path, "a+b")  # noqa: SIM115
    if handle.seek(0, os.SEEK_END) == 0:
        handle.write(b"\0")
        handle.flush()
        os.fsync(handle.fileno())
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    try:
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt  # noqa: PLC0415

                    locking = getattr(msvcrt, "locking")  # noqa: B009
                    locking(handle.fileno(), getattr(msvcrt, "LK_NBLCK"), 1)  # noqa: B009
                else:
                    import fcntl  # noqa: PLC0415

                    flock = getattr(fcntl, "flock")  # noqa: B009
                    flock(  # noqa: B009
                        handle.fileno(),
                        getattr(fcntl, "LOCK_EX") | getattr(fcntl, "LOCK_NB"),  # noqa: B009
                    )
                break
            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    raise MobileTlsConfigurationError(
                        "Timed out securing TLS material generation."
                    ) from None
                time.sleep(0.05)
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt  # noqa: PLC0415

                locking = getattr(msvcrt, "locking")  # noqa: B009
                locking(handle.fileno(), getattr(msvcrt, "LK_UNLCK"), 1)  # noqa: B009
            else:
                import fcntl  # noqa: PLC0415

                flock = getattr(fcntl, "flock")  # noqa: B009
                flock(handle.fileno(), getattr(fcntl, "LOCK_UN"))  # noqa: B009
        finally:
            handle.close()


def _san_for_host(host: str):
    from cryptography import x509  # noqa: PLC0415

    try:
        return x509.IPAddress(ipaddress.ip_address(host))
    except ValueError:
        return x509.DNSName(host)


def _generate_pair(directory: Path, *, certificate_host: str) -> None:
    from cryptography import x509  # noqa: PLC0415
    from cryptography.hazmat.primitives import hashes, serialization  # noqa: PLC0415
    from cryptography.hazmat.primitives.asymmetric import ec  # noqa: PLC0415
    from cryptography.x509.oid import NameOID  # noqa: PLC0415

    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, certificate_host)])
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=CERT_VALIDITY_DAYS))
        .add_extension(
            x509.SubjectAlternativeName([_san_for_host(certificate_host)]), critical=False
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    key_bytes = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    cert_bytes = cert.public_bytes(serialization.Encoding.PEM)
    suffix = f".tmp-{os.getpid()}"
    key_tmp = directory / f"{KEY_FILENAME}{suffix}"
    cert_tmp = directory / f"{CERT_FILENAME}{suffix}"
    try:
        fd = os.open(key_tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, key_bytes)
            os.fsync(fd)
        finally:
            os.close(fd)
        _secure_private_key(key_tmp)
        with open(cert_tmp, "xb") as handle:
            handle.write(cert_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(key_tmp, directory / KEY_FILENAME)
        os.replace(cert_tmp, directory / CERT_FILENAME)
        _secure_private_key(directory / KEY_FILENAME)
    finally:
        key_tmp.unlink(missing_ok=True)
        cert_tmp.unlink(missing_ok=True)


def _load_material(cert_path: Path, key_path: Path, *, certificate_host: str) -> TlsMaterial:
    from cryptography import x509  # noqa: PLC0415
    from cryptography.hazmat.primitives import serialization  # noqa: PLC0415

    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        now = datetime.now(UTC)
        not_before = cert.not_valid_before_utc
        not_after = cert.not_valid_after_utc
        if now < not_before or now >= not_after:
            raise ValueError("certificate outside validity window")
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        expected = (
            ipaddress.ip_address(certificate_host)
            if _looks_like_ip(certificate_host)
            else certificate_host
        )
        values = san.get_values_for_type(
            x509.IPAddress if _looks_like_ip(certificate_host) else x509.DNSName
        )
        if expected not in values:
            raise ValueError("SAN mismatch")
        _secure_private_key(key_path)
        der = cert.public_bytes(serialization.Encoding.DER)
        spki = cert.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        material = TlsMaterial(
            cert_path,
            key_path,
            certificate_host,
            hashlib.sha256(der).hexdigest(),
            base64.b64encode(hashlib.sha256(spki).digest()).decode("ascii"),
        )
        material.build_ssl_context()
        return material
    except MobileTlsConfigurationError:
        raise
    except Exception as exc:
        raise MobileTlsConfigurationError(
            "Existing TLS certificate material is invalid; reset and re-pair explicitly."
        ) from exc


def _looks_like_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def ensure_server_tls_material(
    data_dir: Path | str, *, certificate_host: str = "askrex-desktop"
) -> TlsMaterial:
    host = _canonical_host(certificate_host)
    directory = _tls_dir(Path(data_dir))
    cert_path, key_path = directory / CERT_FILENAME, directory / KEY_FILENAME
    with _generation_lock(directory):
        cert_exists, key_exists = cert_path.exists(), key_path.exists()
        if cert_exists != key_exists:
            raise MobileTlsConfigurationError(
                "TLS certificate material is incomplete; reset and re-pair explicitly."
            )
        if not cert_exists:
            try:
                _generate_pair(directory, certificate_host=host)
            except MobileTlsConfigurationError:
                raise
            except Exception as exc:
                raise MobileTlsConfigurationError(
                    "TLS certificate material could not be provisioned."
                ) from exc
        return _load_material(cert_path, key_path, certificate_host=host)


def resolve_mobile_tls(
    *, host: str, require_tls: bool, data_dir: Path | str, advertised_host: str | None = None
) -> TlsMaterial | None:
    if host_is_loopback(host) and not require_tls:
        return None
    certificate_host = _canonical_host(advertised_host or host)
    if host_is_loopback(certificate_host) and not host_is_loopback(host):
        raise MobileTlsConfigurationError("A LAN bind cannot advertise a loopback host.")
    return ensure_server_tls_material(data_dir, certificate_host=certificate_host)


def clear_tls_material(data_dir: Path | str) -> None:
    directory = _tls_dir(Path(data_dir))
    with _generation_lock(directory):
        (directory / CERT_FILENAME).unlink(missing_ok=True)
        (directory / KEY_FILENAME).unlink(missing_ok=True)


__all__ = [
    "CERT_FILENAME",
    "KEY_FILENAME",
    "TLS_SUBDIR",
    "MobileTlsConfigurationError",
    "TlsMaterial",
    "TransportBinding",
    "advertised_server_url",
    "clear_tls_material",
    "ensure_server_tls_material",
    "host_is_loopback",
    "host_is_wildcard",
    "resolve_mobile_tls",
]
