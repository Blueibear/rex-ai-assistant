"""Tests for US-111: Startup config validation with fail-fast.

Acceptance criteria:
- a config validation step runs before any other initialization
- missing required environment variables produce a specific error message
  naming the missing variable and exit code 1
- invalid values (e.g., non-numeric port, malformed URL) produce a
  descriptive error and exit code 1
- optional variables with defaults do not cause startup failure
- Typecheck passes
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rex.startup_validation import (
    EnvVarSpec,
    _validate_int,
    _validate_url,
    check_startup_env,
    validate_startup_env,
)

# ---------------------------------------------------------------------------
# _validate_int
# ---------------------------------------------------------------------------


class TestValidateInt:
    def test_valid_integer_returns_none(self) -> None:
        assert _validate_int("8080") is None

    def test_zero_returns_none(self) -> None:
        assert _validate_int("0") is None

    def test_negative_integer_returns_none(self) -> None:
        assert _validate_int("-1") is None

    def test_non_numeric_returns_error(self) -> None:
        assert _validate_int("abc") is not None

    def test_float_string_returns_error(self) -> None:
        assert _validate_int("3.14") is not None

    def test_empty_string_returns_error(self) -> None:
        assert _validate_int("") is not None

    def test_error_contains_value(self) -> None:
        err = _validate_int("badport")
        assert err is not None
        assert "badport" in err


# ---------------------------------------------------------------------------
# _validate_url
# ---------------------------------------------------------------------------


class TestValidateUrl:
    def test_http_url_returns_none(self) -> None:
        assert _validate_url("http://localhost:8080") is None

    def test_https_url_returns_none(self) -> None:
        assert _validate_url("https://example.com") is None

    def test_bare_string_returns_error(self) -> None:
        assert _validate_url("not-a-url") is not None

    def test_ftp_scheme_returns_error(self) -> None:
        assert _validate_url("ftp://files.example.com") is not None

    def test_missing_host_returns_error(self) -> None:
        assert _validate_url("http://") is not None

    def test_error_contains_value(self) -> None:
        err = _validate_url("badurl")
        assert err is not None
        assert "badurl" in err


# ---------------------------------------------------------------------------
# validate_startup_env
# ---------------------------------------------------------------------------


class TestValidateStartupEnv:
    def test_required_missing_produces_error(self) -> None:
        specs = [EnvVarSpec("MY_REQUIRED_VAR", required=True)]
        errors = validate_startup_env(specs, env={})
        assert len(errors) == 1

    def test_required_missing_error_names_variable(self) -> None:
        specs = [EnvVarSpec("MY_REQUIRED_VAR", required=True)]
        errors = validate_startup_env(specs, env={})
        assert "MY_REQUIRED_VAR" in errors[0]

    def test_required_present_no_error(self) -> None:
        specs = [EnvVarSpec("MY_REQUIRED_VAR", required=True)]
        errors = validate_startup_env(specs, env={"MY_REQUIRED_VAR": "value"})
        assert errors == []

    def test_optional_missing_no_error(self) -> None:
        specs = [EnvVarSpec("MY_OPTIONAL_VAR", required=False)]
        errors = validate_startup_env(specs, env={})
        assert errors == []

    def test_optional_present_valid_no_error(self) -> None:
        specs = [EnvVarSpec("MY_PORT", required=False, validator=_validate_int)]
        errors = validate_startup_env(specs, env={"MY_PORT": "9000"})
        assert errors == []

    def test_invalid_int_produces_error(self) -> None:
        specs = [EnvVarSpec("MY_PORT", required=False, validator=_validate_int)]
        errors = validate_startup_env(specs, env={"MY_PORT": "not-a-number"})
        assert len(errors) == 1

    def test_invalid_int_error_names_variable(self) -> None:
        specs = [EnvVarSpec("MY_PORT", required=False, validator=_validate_int)]
        errors = validate_startup_env(specs, env={"MY_PORT": "bad"})
        assert "MY_PORT" in errors[0]

    def test_invalid_url_produces_error(self) -> None:
        specs = [EnvVarSpec("MY_URL", required=False, validator=_validate_url)]
        errors = validate_startup_env(specs, env={"MY_URL": "not-a-url"})
        assert len(errors) == 1

    def test_invalid_url_error_names_variable(self) -> None:
        specs = [EnvVarSpec("MY_URL", required=False, validator=_validate_url)]
        errors = validate_startup_env(specs, env={"MY_URL": "bad"})
        assert "MY_URL" in errors[0]

    def test_valid_url_no_error(self) -> None:
        specs = [EnvVarSpec("MY_URL", required=False, validator=_validate_url)]
        errors = validate_startup_env(specs, env={"MY_URL": "https://api.example.com"})
        assert errors == []

    def test_multiple_errors_collected(self) -> None:
        specs = [
            EnvVarSpec("MISSING_REQ", required=True),
            EnvVarSpec("BAD_PORT", required=False, validator=_validate_int),
        ]
        errors = validate_startup_env(specs, env={"BAD_PORT": "abc"})
        assert len(errors) == 2

    def test_empty_specs_no_errors(self) -> None:
        errors = validate_startup_env([], env={})
        assert errors == []

    def test_no_validator_present_value_no_error(self) -> None:
        specs = [EnvVarSpec("SOME_VAR")]
        errors = validate_startup_env(specs, env={"SOME_VAR": "anything"})
        assert errors == []

    def test_missing_optional_with_no_validator_no_error(self) -> None:
        specs = [EnvVarSpec("SOME_VAR")]
        errors = validate_startup_env(specs, env={})
        assert errors == []

    def test_uses_os_environ_when_env_not_provided(self) -> None:
        specs = [EnvVarSpec("_TEST_US111_MISSING_REQ", required=True)]
        with patch.dict("os.environ", {}, clear=False):
            # ensure the var is absent
            import os

            os.environ.pop("_TEST_US111_MISSING_REQ", None)
            errors = validate_startup_env(specs)
        assert len(errors) == 1
        assert "_TEST_US111_MISSING_REQ" in errors[0]


# ---------------------------------------------------------------------------
# check_startup_env
# ---------------------------------------------------------------------------


class TestCheckStartupEnv:
    def test_exits_with_code_1_on_missing_required(self) -> None:
        specs = [EnvVarSpec("MISSING_REQ", required=True)]
        with pytest.raises(SystemExit) as exc_info:
            check_startup_env(specs, env={})
        assert exc_info.value.code == 1

    def test_exits_with_code_1_on_invalid_value(self) -> None:
        specs = [EnvVarSpec("BAD_PORT", required=False, validator=_validate_int)]
        with pytest.raises(SystemExit) as exc_info:
            check_startup_env(specs, env={"BAD_PORT": "not-an-int"})
        assert exc_info.value.code == 1

    def test_no_exit_when_no_errors(self) -> None:
        errors = check_startup_env([], env={})
        assert errors == []

    def test_error_messages_printed_to_stderr(self, capsys: pytest.CaptureFixture) -> None:
        specs = [EnvVarSpec("MISSING_SECRET", required=True)]
        with pytest.raises(SystemExit):
            check_startup_env(specs, env={})
        captured = capsys.readouterr()
        assert "MISSING_SECRET" in captured.err

    def test_exit_on_error_false_returns_errors(self) -> None:
        specs = [EnvVarSpec("MISSING_REQ", required=True)]
        errors = check_startup_env(specs, env={}, exit_on_error=False)
        assert len(errors) == 1
        assert "MISSING_REQ" in errors[0]

    def test_exit_on_error_false_does_not_exit(self) -> None:
        specs = [EnvVarSpec("MISSING_REQ", required=True)]
        # Should NOT raise SystemExit
        errors = check_startup_env(specs, env={}, exit_on_error=False)
        assert errors  # errors present but no exit

    def test_returns_empty_list_when_all_valid(self) -> None:
        specs = [
            EnvVarSpec("MY_PORT", required=False, validator=_validate_int),
            EnvVarSpec("MY_URL", required=False, validator=_validate_url),
        ]
        env = {"MY_PORT": "8080", "MY_URL": "https://api.example.com"}
        errors = check_startup_env(specs, env=env)
        assert errors == []


# ---------------------------------------------------------------------------
# Default _ENV_SPECS (built-in known env vars)
# ---------------------------------------------------------------------------


class TestDefaultEnvSpecs:
    def test_rex_speak_port_validates_as_int(self) -> None:
        from rex.startup_validation import _ENV_SPECS

        port_spec = next((s for s in _ENV_SPECS if s.name == "REX_SPEAK_PORT"), None)
        assert port_spec is not None
        assert port_spec.validator is not None
        assert port_spec.validator("abc") is not None
        assert port_spec.validator("5005") is None

    def test_openai_base_url_validates_as_url(self) -> None:
        from rex.startup_validation import _ENV_SPECS

        url_spec = next((s for s in _ENV_SPECS if s.name == "OPENAI_BASE_URL"), None)
        assert url_spec is not None
        assert url_spec.validator is not None
        assert url_spec.validator("not-a-url") is not None
        assert url_spec.validator("https://api.openai.com/v1") is None

    def test_no_required_vars_in_defaults(self) -> None:
        """By default no env vars are required (secrets are optional)."""
        from rex.startup_validation import _ENV_SPECS

        required = [s for s in _ENV_SPECS if s.required]
        assert required == []
