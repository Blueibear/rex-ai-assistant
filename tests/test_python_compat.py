from python_compat import (
    DEFAULT_INSTALL_LABEL,
    SUPPORTED_VERSION_LABEL,
    is_supported_python,
    unsupported_python_message,
)


def test_python_compat_accepts_python_311() -> None:
    assert is_supported_python((3, 11, 0))
    assert is_supported_python((3, 11, 9))


def test_python_compat_rejects_other_minor_versions() -> None:
    assert not is_supported_python((3, 10, 14))
    assert not is_supported_python((3, 12, 0))
    assert not is_supported_python((3, 13, 2))


def test_python_compat_message_is_actionable() -> None:
    message = unsupported_python_message((3, 13, 1), install_target=DEFAULT_INSTALL_LABEL)

    assert "3.13.1" in message
    assert DEFAULT_INSTALL_LABEL in message
    assert SUPPORTED_VERSION_LABEL in message
    assert "known to fail" in message
