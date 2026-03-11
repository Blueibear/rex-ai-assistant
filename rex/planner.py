"""Planner for Rex - generates multi-step workflows from high-level goals.

This module implements a rule-based planner that can parse natural language goals
and generate structured workflows with appropriate tools, risk levels, and
idempotency keys.

The planner:
1. Parses goals into sub-tasks using pattern matching
2. Selects appropriate tools from the ToolRegistry
3. Assigns risk levels and idempotency keys
4. Validates plans against the PolicyEngine

Current limitations:
- Rule-based parser with limited patterns (not LLM-based)
- Only handles predefined goal patterns
- No dynamic plan optimization or learning

Usage:
    from rex.planner import Planner
    from rex.tool_registry import get_tool_registry
    from rex.policy_engine import get_policy_engine

    planner = Planner(
        tool_registry=get_tool_registry(),
        policy_engine=get_policy_engine()
    )
    workflow = planner.plan("send monthly newsletter")
    if planner.validate_workflow(workflow):
        workflow.save()
"""

from __future__ import annotations

import logging
import re
from typing import Any

from rex.contracts import ToolCall
from rex.policy_engine import PolicyEngine, get_policy_engine
from rex.tool_registry import ToolRegistry, get_tool_registry
from rex.workflow import Workflow, WorkflowStep, generate_step_id, generate_workflow_id

logger = logging.getLogger(__name__)


class PlannerError(Exception):
    """Base exception for planner errors."""

    pass


class UnableToPlanError(PlannerError):
    """Raised when the planner cannot parse or plan for a goal."""

    pass


class InvalidWorkflowError(PlannerError):
    """Raised when a workflow fails validation."""

    pass


class Planner:
    """Planner that generates workflows from high-level goals.

    The planner uses a rule-based approach to parse goals and select tools.
    It validates plans against the policy engine to ensure all steps are allowed.

    Attributes:
        tool_registry: Registry of available tools
        policy_engine: Policy engine for validation
    """

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        policy_engine: PolicyEngine | None = None,
    ):
        """Initialize the planner.

        Args:
            tool_registry: Tool registry to use. Defaults to global instance.
            policy_engine: Policy engine to use. Defaults to global instance.
        """
        self.tool_registry = tool_registry or get_tool_registry()
        self.policy_engine = policy_engine or get_policy_engine()

        # Define planning rules: pattern -> step generator function
        self._rules: list[tuple[re.Pattern, Any]] = [
            # Email patterns
            (
                re.compile(r"send\s+(?:monthly|weekly|daily)?\s*newsletter", re.IGNORECASE),
                self._plan_newsletter,
            ),
            (
                re.compile(r"send\s+(?:an?\s+)?(?:\w+\s+)*email(?:\s+to\s+(.+))?", re.IGNORECASE),
                self._plan_send_email,
            ),
            (re.compile(r"email\s+(.+)", re.IGNORECASE), self._plan_send_email),
            # Calendar patterns
            (
                re.compile(r"schedule\s+(?:a\s+)?(?:meeting|event)\s+(.+)", re.IGNORECASE),
                self._plan_schedule_event,
            ),
            (
                re.compile(r"create\s+(?:a\s+)?calendar\s+event\s+(.+)", re.IGNORECASE),
                self._plan_schedule_event,
            ),
            # Weather patterns
            (
                re.compile(
                    r"(?:check|get|what'?s)\s+(?:the\s+)?weather(?:\s+(?:in|for)\s+(.+))?",
                    re.IGNORECASE,
                ),
                self._plan_check_weather,
            ),
            # Time patterns
            (
                re.compile(
                    r"(?:check|get|what'?s)\s+(?:the\s+)?(?:current\s+)?time(?:\s+(?:in|at)\s+(.+))?",
                    re.IGNORECASE,
                ),
                self._plan_check_time,
            ),
            # Home automation patterns
            (
                re.compile(r"turn\s+(on|off)\s+(?:the\s+)?(.+)", re.IGNORECASE),
                self._plan_home_control,
            ),
            (re.compile(r"set\s+(.+?)\s+to\s+(.+)", re.IGNORECASE), self._plan_home_control),
            # Web search patterns
            (
                re.compile(r"(?:search|look\s+up|find)\s+(?:for\s+)?(.+)", re.IGNORECASE),
                self._plan_web_search,
            ),
            # Report generation patterns
            (
                re.compile(
                    r"(?:create|generate|send)\s+(?:monthly|weekly|daily)\s+report", re.IGNORECASE
                ),
                self._plan_report,
            ),
        ]

    def plan(self, goal: str, requested_by: str | None = None) -> Workflow:
        """Generate a workflow from a high-level goal.

        Args:
            goal: Natural language description of the goal
            requested_by: Who requested this plan (for audit trail)

        Returns:
            A Workflow with steps to accomplish the goal

        Raises:
            UnableToPlanError: If the goal cannot be parsed or planned
        """
        logger.info("Planning workflow for goal: %s", goal)

        goal = goal.strip()
        if not goal:
            raise UnableToPlanError("Empty goal provided")

        # Try to match against known patterns
        for pattern, step_generator in self._rules:
            match = pattern.search(goal)
            if match:
                try:
                    steps = step_generator(match, goal)
                    if steps:
                        workflow = Workflow(
                            workflow_id=generate_workflow_id(),
                            title=goal,
                            steps=steps,
                            requested_by=requested_by or "planner",
                        )
                        logger.info(
                            "Generated workflow %s with %d steps", workflow.workflow_id, len(steps)
                        )
                        return workflow
                except Exception as e:
                    logger.warning("Step generator failed for pattern %s: %s", pattern.pattern, e)
                    continue

        # No pattern matched
        raise UnableToPlanError(
            f"Unable to plan for goal: '{goal}'. "
            "The planner does not recognize this goal pattern. "
            "Supported patterns include: send email, check weather, "
            "schedule event, turn on/off devices, search web, etc."
        )

    def validate_workflow(self, workflow: Workflow) -> bool:
        """Validate a workflow against tool availability and policies.

        This checks:
        1. All tools in the workflow exist in the registry
        2. All tools pass policy checks (not denied)
        3. Required credentials are available

        Args:
            workflow: Workflow to validate

        Returns:
            True if the workflow is valid, False otherwise
        """
        logger.info("Validating workflow %s", workflow.workflow_id)

        for step in workflow.steps:
            if step.tool_call is None:
                continue

            tool_name = step.tool_call.tool

            # Check tool exists in registry
            if not self.tool_registry.has_tool(tool_name):
                logger.error("Validation failed: Tool '%s' not found in registry", tool_name)
                return False

            # Check credentials are available
            try:
                self.tool_registry.validate_credentials_for_tool(tool_name)
            except Exception as e:
                logger.error("Validation failed: Credentials missing for '%s': %s", tool_name, e)
                return False

            # Check policy allows the tool (not denied)
            decision = self.policy_engine.decide(step.tool_call, metadata={})
            if decision.denied:
                logger.error(
                    "Validation failed: Policy denies tool '%s': %s", tool_name, decision.reason
                )
                return False

            # Set requires_approval flag based on policy
            if decision.requires_approval:
                step.requires_approval = True

        logger.info("Workflow %s validation passed", workflow.workflow_id)
        return True

    # --- Step generator methods ---

    def _plan_newsletter(self, match: re.Match, goal: str) -> list[WorkflowStep]:
        """Generate steps for sending a newsletter."""
        # For newsletter: 1) search for contacts, 2) compose message, 3) send email
        steps = []

        # Step 1: Search for newsletter contacts (simulated with web_search)
        if self.tool_registry.has_tool("web_search"):
            steps.append(
                WorkflowStep(
                    step_id=generate_step_id(),
                    description="Search for newsletter contact list",
                    tool_call=ToolCall(
                        tool="web_search",
                        args={"query": "newsletter contacts"},
                    ),
                    idempotency_key=f"newsletter_search_{goal}",
                )
            )

        # Step 2: Send newsletter email
        if self.tool_registry.has_tool("send_email"):
            steps.append(
                WorkflowStep(
                    step_id=generate_step_id(),
                    description="Send newsletter email",
                    tool_call=ToolCall(
                        tool="send_email",
                        args={
                            "to": "newsletter@example.com",
                            "subject": "Newsletter",
                            "body": "Newsletter content",
                        },
                    ),
                    idempotency_key=f"newsletter_send_{goal}",
                )
            )

        return steps

    def _plan_send_email(self, match: re.Match, goal: str) -> list[WorkflowStep]:
        """Generate steps for sending an email."""
        recipient = "recipient@example.com"
        if match.lastindex and match.lastindex >= 1 and match.group(1):
            recipient = match.group(1).strip()

        if not self.tool_registry.has_tool("send_email"):
            return []

        return [
            WorkflowStep(
                step_id=generate_step_id(),
                description=f"Send email to {recipient}",
                tool_call=ToolCall(
                    tool="send_email",
                    args={
                        "to": recipient,
                        "subject": "Message from Rex",
                        "body": goal,
                    },
                ),
                idempotency_key=f"email_{recipient}_{goal[:20]}",
            )
        ]

    def _plan_schedule_event(self, match: re.Match, goal: str) -> list[WorkflowStep]:
        """Generate steps for scheduling a calendar event."""
        event_details = match.group(1) if match.lastindex and match.lastindex >= 1 else "event"
        event_details = event_details.strip()

        if not self.tool_registry.has_tool("calendar_create_event"):
            return []

        return [
            WorkflowStep(
                step_id=generate_step_id(),
                description=f"Create calendar event: {event_details}",
                tool_call=ToolCall(
                    tool="calendar_create_event",
                    args={
                        "summary": event_details,
                        "description": goal,
                    },
                ),
                idempotency_key=f"calendar_{event_details[:20]}",
            )
        ]

    def _plan_check_weather(self, match: re.Match, goal: str) -> list[WorkflowStep]:
        """Generate steps for checking weather."""
        location = (
            match.group(1) if match.lastindex and match.lastindex >= 1 else "current location"
        )
        if location:
            location = location.strip()
        else:
            location = "current location"

        if not self.tool_registry.has_tool("weather_now"):
            return []

        return [
            WorkflowStep(
                step_id=generate_step_id(),
                description=f"Check weather in {location}",
                tool_call=ToolCall(
                    tool="weather_now",
                    args={"location": location},
                ),
                idempotency_key=f"weather_{location}",
            )
        ]

    def _plan_check_time(self, match: re.Match, goal: str) -> list[WorkflowStep]:
        """Generate steps for checking time."""
        location = match.group(1) if match.lastindex and match.lastindex >= 1 else "local"
        if location:
            location = location.strip()
        else:
            location = "local"

        if not self.tool_registry.has_tool("time_now"):
            return []

        return [
            WorkflowStep(
                step_id=generate_step_id(),
                description=f"Get current time in {location}",
                tool_call=ToolCall(
                    tool="time_now",
                    args={"location": location},
                ),
                idempotency_key=f"time_{location}",
            )
        ]

    def _plan_home_control(self, match: re.Match, goal: str) -> list[WorkflowStep]:
        """Generate steps for home automation control."""
        if not self.tool_registry.has_tool("home_assistant_call_service"):
            return []

        # Try to extract action (on/off) and device name
        action = match.group(1) if match.lastindex and match.lastindex >= 1 else "on"
        device = match.group(2) if match.lastindex and match.lastindex >= 2 else "device"

        if action and device:
            action = action.strip().lower()
            device = device.strip()

        return [
            WorkflowStep(
                step_id=generate_step_id(),
                description=f"Turn {action} {device}",
                tool_call=ToolCall(
                    tool="home_assistant_call_service",
                    args={
                        "domain": "light" if "light" in device.lower() else "switch",
                        "service": "turn_on" if action == "on" else "turn_off",
                        "entity_id": (
                            f"light.{device.replace(' ', '_')}"
                            if "light" in device.lower()
                            else f"switch.{device.replace(' ', '_')}"
                        ),
                    },
                ),
                idempotency_key=f"home_{action}_{device}",
            )
        ]

    def _plan_web_search(self, match: re.Match, goal: str) -> list[WorkflowStep]:
        """Generate steps for web search."""
        query = match.group(1) if match.lastindex and match.lastindex >= 1 else goal
        query = query.strip()

        if not self.tool_registry.has_tool("web_search"):
            return []

        return [
            WorkflowStep(
                step_id=generate_step_id(),
                description=f"Search for: {query}",
                tool_call=ToolCall(
                    tool="web_search",
                    args={"query": query},
                ),
                idempotency_key=f"search_{query[:30]}",
            )
        ]

    def select_tool(self, candidates: list[str]) -> str | None:
        """Select the most appropriate tool from a list of candidates.

        Iterates through the candidates in priority order and returns the first
        tool that is registered, enabled, and has all credentials available.

        Args:
            candidates: Ordered list of tool names to consider, from most
                preferred to least preferred.

        Returns:
            The name of the first suitable tool, or None if no candidate is
            available.
        """
        for tool_name in candidates:
            if not self.tool_registry.has_tool(tool_name):
                logger.debug("select_tool: '%s' not registered, skipping", tool_name)
                continue

            meta = self.tool_registry.get_tool(tool_name)
            if meta is not None and not meta.enabled:
                logger.debug("select_tool: '%s' is disabled, skipping", tool_name)
                continue

            all_available, missing = self.tool_registry.check_credentials(tool_name)
            if not all_available:
                logger.debug(
                    "select_tool: '%s' missing credentials %s, skipping", tool_name, missing
                )
                continue

            logger.debug("select_tool: selected '%s'", tool_name)
            return tool_name

        logger.warning("select_tool: no suitable tool found among %s", candidates)
        return None

    def build_prompt(self, goal: str) -> str:
        """Build a prompt for LLM-assisted planning.

        Includes the task description and a list of available tools so the LLM
        has full context for reasoning.

        Args:
            goal: Natural language description of the goal.

        Returns:
            A formatted prompt string containing the task and tool list.
        """
        tools = self.tool_registry.list_tools()
        if tools:
            tool_lines = "\n".join(
                f"- {t.name}: {t.description}" for t in tools
            )
            tools_section = f"Available tools:\n{tool_lines}"
        else:
            tools_section = "Available tools: none"

        return (
            f"Task: {goal}\n\n"
            f"{tools_section}\n\n"
            "Plan the steps needed to accomplish this task using the available tools."
        )

    def plan_with_fallback(
        self,
        goal: str,
        fallback_goals: list[str] | None = None,
        requested_by: str | None = None,
    ) -> Workflow:
        """Generate a workflow, falling back to alternate goals on failure.

        Tries to plan the primary ``goal`` first.  If planning fails (e.g. no
        matching pattern or no tools available), each entry in
        ``fallback_goals`` is tried in order.  All failures are logged.

        Args:
            goal: Primary natural-language goal.
            fallback_goals: Ordered list of alternate goals to try when the
                primary goal cannot be planned.
            requested_by: Who requested the plan (for audit trail).

        Returns:
            A Workflow for the first goal that could be planned successfully.

        Raises:
            UnableToPlanError: If the primary goal and all fallbacks fail.
        """
        candidates = [goal] + list(fallback_goals or [])
        last_error: Exception | None = None

        for attempt, candidate in enumerate(candidates):
            try:
                workflow = self.plan(candidate, requested_by=requested_by)
                if attempt > 0:
                    logger.info(
                        "plan_with_fallback: succeeded on fallback #%d ('%s')",
                        attempt,
                        candidate,
                    )
                return workflow
            except PlannerError as exc:
                logger.warning(
                    "plan_with_fallback: attempt %d failed for '%s': %s",
                    attempt,
                    candidate,
                    exc,
                )
                last_error = exc

        raise UnableToPlanError(
            f"All planning attempts failed. Primary goal: '{goal}'. "
            f"Fallbacks tried: {fallback_goals}. "
            f"Last error: {last_error}"
        )

    def execute_step_with_fallback(
        self,
        primary_step: WorkflowStep,
        fallback_steps: list[WorkflowStep],
        executor: Any,
    ) -> Any:
        """Execute a workflow step, falling back to alternates on failure.

        Calls ``executor(step)`` with the primary step.  If it raises an
        exception, each fallback step is tried in order.  All failures are
        logged so the caller has a full audit trail.

        Args:
            primary_step: The preferred step to execute first.
            fallback_steps: Ordered list of alternate steps to try.
            executor: Callable that accepts a WorkflowStep and returns a result.
                May raise any exception to signal failure.

        Returns:
            The result returned by the first successful executor call.

        Raises:
            Exception: Re-raises the last failure if every step fails.
        """
        all_steps = [primary_step] + list(fallback_steps)
        last_exc: Exception | None = None

        for attempt, step in enumerate(all_steps):
            tool_name = step.tool_call.tool if step.tool_call else "<no tool>"
            try:
                result = executor(step)
                if attempt > 0:
                    logger.info(
                        "execute_step_with_fallback: succeeded on fallback #%d (tool='%s')",
                        attempt,
                        tool_name,
                    )
                return result
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "execute_step_with_fallback: attempt %d failed (tool='%s'): %s",
                    attempt,
                    tool_name,
                    exc,
                )
                last_exc = exc

        raise last_exc  # type: ignore[misc]

    def _plan_report(self, match: re.Match, goal: str) -> list[WorkflowStep]:
        """Generate steps for creating and sending a report."""
        steps = []

        # Step 1: Gather data (simulated with web_search)
        if self.tool_registry.has_tool("web_search"):
            steps.append(
                WorkflowStep(
                    step_id=generate_step_id(),
                    description="Gather report data",
                    tool_call=ToolCall(
                        tool="web_search",
                        args={"query": "report metrics data"},
                    ),
                    idempotency_key=f"report_data_{goal}",
                )
            )

        # Step 2: Send report via email
        if self.tool_registry.has_tool("send_email"):
            steps.append(
                WorkflowStep(
                    step_id=generate_step_id(),
                    description="Send report via email",
                    tool_call=ToolCall(
                        tool="send_email",
                        args={
                            "to": "reports@example.com",
                            "subject": "Report",
                            "body": "Report content",
                        },
                    ),
                    idempotency_key=f"report_send_{goal}",
                )
            )

        return steps


__all__ = [
    "Planner",
    "PlannerError",
    "UnableToPlanError",
    "InvalidWorkflowError",
]
