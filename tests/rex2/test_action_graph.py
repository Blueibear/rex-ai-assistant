from __future__ import annotations

import pytest

from rex.actions.graph import ActionGraph, ActionNode, InvalidActionGraph
from rex.tools.execution import ToolOperation


def test_action_graph_captures_dependencies_authority_and_verification_metadata() -> None:
    read = ActionNode(
        action_id="read-weather",
        tool_name="weather_now",
        operation=ToolOperation.READ,
        authorized=True,
        conflict_keys=("weather:dallas",),
    )
    mutate = ActionNode(
        action_id="set-light",
        tool_name="ha_light_set",
        operation=ToolOperation.MUTATION,
        dependencies=("read-weather",),
        authorized=True,
        confirmation_required=True,
        verification_required=True,
        postcondition="light.kitchen == on",
        conflict_keys=("ha:light.kitchen",),
    )
    graph = ActionGraph(plan_id="plan-1", nodes=(read, mutate))

    assert graph.node("set-light") is mutate
    assert graph.dependencies_of("set-light") == ("read-weather",)
    assert mutate.verification_required is True
    assert mutate.postcondition == "light.kitchen == on"
    assert mutate.confirmation_required is True


def test_action_graph_rejects_unknown_dependency_and_cycles() -> None:
    with pytest.raises(InvalidActionGraph, match="unknown dependency"):
        ActionGraph(
            plan_id="unknown-dep",
            nodes=(ActionNode("a", "tool-a", dependencies=("missing",)),),
        )

    with pytest.raises(InvalidActionGraph, match="cycle"):
        ActionGraph(
            plan_id="cycle",
            nodes=(
                ActionNode("a", "tool-a", dependencies=("b",)),
                ActionNode("b", "tool-b", dependencies=("a",)),
            ),
        )


def test_action_node_arguments_are_immutable_after_validation() -> None:
    node = ActionNode(
        "immutable-args",
        "weather_now",
        args={"location": "Dallas"},
    )

    with pytest.raises(TypeError):
        node.args["location"] = "Austin"  # type: ignore[index]

    assert dict(node.args) == {"location": "Dallas"}
