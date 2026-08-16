from __future__ import annotations

from rex.tools.execution import ToolExecutionLifecycle, ToolOutcome
from rex.tools.registry import Tool


def _tool(**overrides) -> Tool:
    values = {
        "name": "example_tool",
        "description": "Example tool",
        "capability_tags": ["example"],
        "requires_config": [],
        "handler": lambda **kwargs: "ok",
    }
    values.update(overrides)
    return Tool(**values)


def test_unavailable_tool_names_exact_missing_config_and_retry() -> None:
    result = ToolExecutionLifecycle().execute(
        _tool(requires_config=["example_api_key"]), {}, available=False
    )

    assert result.status == ToolOutcome.UNAVAILABLE.value
    assert result.error is not None
    assert "example_api_key" in result.error
    assert "Required Rex config key" in result.error
    assert "existing settings/credential source" in result.error
    assert "then retry" in result.error.lower()


def test_missing_argument_error_tells_user_what_to_provide_next() -> None:
    result = ToolExecutionLifecycle().execute(_tool(required_args=("query",)), {}, available=True)

    assert result.status == ToolOutcome.DENIED.value
    assert result.error is not None
    assert "query" in result.error
    assert "Provide the missing value" in result.error


def test_missing_permission_error_explains_access_request() -> None:
    result = ToolExecutionLifecycle().execute(
        _tool(required_permissions=("files.private.read",), requires_identity=True),
        {},
        context={"user_id": "james", "granted_permissions": set()},
        available=True,
    )

    assert result.status == ToolOutcome.DENIED.value
    assert result.error is not None
    assert "files.private.read" in result.error
    assert "Ask an administrator" in result.error
    assert "then retry" in result.error.lower()


def test_unavailable_tool_reports_only_missing_config_keys() -> None:
    config = type("Config", (), {"api_host": "configured", "api_key": ""})()
    result = ToolExecutionLifecycle().execute(
        _tool(requires_config=["api_host", "api_key"]),
        {},
        available=False,
        runtime_config=config,
    )

    assert result.error is not None
    assert "api_key" in result.error
    assert "api_host" not in result.error


def test_permission_error_reports_only_missing_permissions() -> None:
    result = ToolExecutionLifecycle().execute(
        _tool(
            required_permissions=("files.read", "files.write"),
            requires_identity=True,
        ),
        {},
        context={"user_id": "james", "granted_permissions": {"files.read"}},
        available=True,
    )

    assert result.error is not None
    assert "files.write" in result.error
    assert "files.read" not in result.error
