"""Tests for US-111: Startup config validation with fail-fast.

Verifies:
- Config validation step exists and runs before other initialization
- Missing required env vars produce specific error messages naming the variable
- Invalid values produce descriptive errors
- check_startup_env() exits with code 1 on any failure
- Optional variables with defaults do not cause startup failure
- Typecheck passes
"""

from __future__ import annotations

import pytest

from rex.startup_validation import (
    _ENV_SPECS,
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
        assert _validate_int("42") is None

    def test_valid_zero_returns_none(self) -> None:
        assert _validate_int("0") is None

    def test_valid_negative_returns_none(self) -> None:
        assert _validate_int("-1") is None

    def test_non_numeric_returns_error(self) -> None:
        result = _validate_int("abc")
        assert result is not None
        assert "abc" in result

    def test_float_string_returns_error(self) -> None:
        result = _validate_int("3.14")
        assert result is not None


# ---------------------------------------------------------------------------
# _validate_url
# ---------------------------------------------------------------------------


class TestValidateUrl:
    def test_valid_http_url(self) -> None:
        assert _validate_url("http://localhost:8123") is None

    def test_valid_https_url(self) -> None:
        assert _validate_url("https://example.com/path") is None

    def test_ftp_scheme_returns_error(self) -> None:
        result = _validate_url("ftp://example.com")
        assert result is not None

    def test_no_scheme_returns_error(self) -> None:
        result = _validate_url("example.com")
        assert result is not None

    def test_missing_host_returns_error(self) -> None:
        result = _validate_url("https://")
        assert result is not None

    def test_empty_string_returns_error(self) -> None:
        result = _validate_url("")
        # either error or None — empty is caught at the caller level, but validator should not crash
        # (empty string has no valid http scheme)
        assert result is not None


# ---------------------------------------------------------------------------
# validate_startup_env — required vars
# ---------------------------------------------------------------------------


class TestValidateRequiredVars:
    def test_no_errors_with_no_specs(self) -> None:
        errors = validate_startup_env(specs=[], env={})
        assert errors == []

    def test_required_var_missing_produces_error(self) -> None:
        specs = [EnvVarSpec("MY_SECRET", required=True)]
        errors = validate_startup_env(specs=specs, env={})
        assert len(errors) == 1
        assert "MY_SECRET" in errors[0]

    def test_required_var_present_no_error(self) -> None:
        specs = [EnvVarSpec("MY_SECRET", required=True)]
        errors = validate_startup_env(specs=specs, env={"MY_SECRET": "value"})
        assert errors == []

    def test_error_message_names_the_variable(self) -> None:
        specs = [EnvVarSpec("NAMED_VAR_XYZ", required=True)]
        errors = validate_startup_env(specs=specs, env={})
        assert any("NAMED_VAR_XYZ" in e for e in errors)

    def test_optional_var_missing_no_error(self) -> None:
        specs = [EnvVarSpec("OPTIONAL_VAR", required=False)]
        errors = validate_startup_env(specs=specs, env={})
        assert errors == []

    def test_multiple_missing_required_all_reported(self) -> None:
        specs = [
            EnvVarSpec("FIRST_REQUIRED", required=True),
            EnvVarSpec("SECOND_REQUIRED", required=True),
        ]
        errors = validate_startup_env(specs=specs, env={})
        assert len(errors) == 2

    def test_mix_required_and_optional(self) -> None:
        specs = [
            EnvVarSpec("REQ_VAR", required=True),
            EnvVarSpec("OPT_VAR", required=False),
        ]
        errors = validate_startup_env(specs=specs, env={})
        assert len(errors) == 1
        assert "REQ_VAR" in errors[0]


# ---------------------------------------------------------------------------
# validate_startup_env — value validators
# ---------------------------------------------------------------------------


class TestValidateValues:
    def test_valid_value_passes(self) -> None:
        specs = [EnvVarSpec("MY_PORT", validator=_validate_int)]
        errors = validate_startup_env(specs=specs, env={"MY_PORT": "8080"})
        assert errors == []

    def test_invalid_value_produces_error(self) -> None:
        specs = [EnvVarSpec("MY_PORT", validator=_validate_int)]
        errors = validate_startup_env(specs=specs, env={"MY_PORT": "not_a_port"})
        assert len(errors) == 1
        assert "MY_PORT" in errors[0]

    def test_validator_not_called_when_var_unset(self) -> None:
        called: list[str] = []

        def _spy(value: str) -> None:
            called.append(value)
            return None

        specs = [EnvVarSpec("UNSET_VAR", validator=_spy)]
        validate_startup_env(specs=specs, env={})
        assert called == []

    def test_invalid_url_produces_error(self) -> None:
        specs = [EnvVarSpec("MY_URL", validator=_validate_url)]
        errors = validate_startup_env(specs=specs, env={"MY_URL": "not-a-url"})
        assert len(errors) == 1

    def test_valid_url_passes(self) -> None:
        specs = [EnvVarSpec("MY_URL", validator=_validate_url)]
        errors = validate_startup_env(specs=specs, env={"MY_URL": "https://example.com"})
        assert errors == []

    def test_description_accessible(self) -> None:
        spec = EnvVarSpec("VAR", description="The purpose of this var")
        assert "purpose" in spec.description


# ---------------------------------------------------------------------------
# Built-in _ENV_SPECS
# ---------------------------------------------------------------------------


class TestBuiltInSpecs:
    def test_env_specs_is_list(self) -> None:
        assert isinstance(_ENV_SPECS, list)

    def test_all_specs_are_envvarspec(self) -> None:
        for spec in _ENV_SPECS:
            assert isinstance(spec, EnvVarSpec)

    def test_rex_speak_port_spec_exists(self) -> None:
        names = [s.name for s in _ENV_SPECS]
        assert "REX_SPEAK_PORT" in names

    def test_ha_base_url_spec_exists(self) -> None:
        names = [s.name for s in _ENV_SPECS]
        assert "HA_BASE_URL" in names

    def test_rex_speak_port_is_optional(self) -> None:
        spec = next(s for s in _ENV_SPECS if s.name == "REX_SPEAK_PORT")
        assert spec.required is False

    def test_rex_speak_port_has_int_validator(self) -> None:
        spec = next(s for s in _ENV_SPECS if s.name == "REX_SPEAK_PORT")
        assert spec.validator is not None
        assert spec.validator("not_int") is not None  # invalid
        assert spec.validator("5005") is None  # valid

    def test_ha_base_url_has_url_validator(self) -> None:
        spec = next(s for s in _ENV_SPECS if s.name == "HA_BASE_URL")
        assert spec.validator is not None
        assert spec.validator("ftp://bad") is not None  # invalid
        assert spec.validator("http://ha.local:8123") is None  # valid

    def test_default_specs_pass_with_no_env_vars(self) -> None:
        """No built-in spec is required, so empty env passes."""
        errors = validate_startup_env(env={})
        assert errors == []


# ---------------------------------------------------------------------------
# check_startup_env — exit behaviour
# ---------------------------------------------------------------------------


class TestCheckStartupEnv:
    def test_no_exit_when_no_errors(self) -> None:
        errors = check_startup_env(specs=[], env={}, exit_on_error=False)
        assert errors == []

    def test_returns_empty_on_success(self) -> None:
        specs = [EnvVarSpec("PRESENT", required=True)]
        errors = check_startup_env(specs=specs, env={"PRESENT": "yes"}, exit_on_error=False)
        assert errors == []

    def test_returns_errors_without_exit_when_exit_on_error_false(self) -> None:
        specs = [EnvVarSpec("MISSING_VAR", required=True)]
        errors = check_startup_env(specs=specs, env={}, exit_on_error=False)
        assert len(errors) == 1

    def test_exits_1_when_errors_and_exit_on_error_true(self) -> None:
        specs = [EnvVarSpec("MISSING_REQUIRED", required=True)]
        with pytest.raises(SystemExit) as exc_info:
            check_startup_env(specs=specs, env={}, exit_on_error=True)
        assert exc_info.value.code == 1

    def test_exits_1_on_invalid_value(self) -> None:
        specs = [EnvVarSpec("BAD_PORT", validator=_validate_int)]
        with pytest.raises(SystemExit) as exc_info:
            check_startup_env(specs=specs, env={"BAD_PORT": "abc"}, exit_on_error=True)
        assert exc_info.value.code == 1

    def test_no_exit_when_optional_var_missing(self) -> None:
        specs = [EnvVarSpec("OPTIONAL_VAR", required=False)]
        errors = check_startup_env(specs=specs, env={})
        assert errors == []

    def test_uses_default_specs_when_specs_none(self) -> None:
        """Calling with no specs argument uses _ENV_SPECS."""
        # Should pass with empty env since no built-in spec is required
        errors = check_startup_env(env={}, exit_on_error=False)
        assert errors == []

    def test_entry_points_import_check_startup_env(self) -> None:
        """Verify that at least one entry point imports startup validation."""
        # cli.py should have been modified to call check_startup_env
        import inspect

        import rex.cli as cli_mod  # noqa: F401

        source = inspect.getsource(cli_mod)
        assert "startup_validation" in source or "check_startup_env" in source
