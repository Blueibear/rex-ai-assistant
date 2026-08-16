"""Fail-closed capability-gap discovery and structured recovery actions.

US-078 keeps recovery separate from execution authority. A recovery plan may
explain what Rex can use, enable, connect, compose, or ask to build, but it does
not execute any of those actions itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from .registry import Capability, CapabilityRegistry

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "can",
        "could",
        "for",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "that",
        "the",
        "this",
        "to",
        "with",
        "would",
        "you",
    }
)
_DIRECT_MATCH_THRESHOLD = 0.75
_BLOCKED_HEALTH = frozenset({"unhealthy", "unavailable"})
_GENERIC_ACTION_TOKENS = frozenset(
    {
        "book",
        "build",
        "call",
        "connect",
        "control",
        "copy",
        "create",
        "delete",
        "download",
        "enable",
        "find",
        "install",
        "launch",
        "move",
        "open",
        "play",
        "read",
        "remove",
        "schedule",
        "search",
        "send",
        "set",
        "start",
        "stop",
        "text",
        "turn",
        "upload",
        "write",
    }
)
_CAPABILITY_BUILD_RE = re.compile(
    r"(?:\b(?:build|create|add|install|develop|make)\b.{0,80}\b(?:capability|tool|integration|plugin|skill|provider)\b"
    r"|\b(?:capability|tool|integration|plugin|skill|provider)\b.{0,80}\b(?:build|create|add|install|develop|make)\b)",
    re.IGNORECASE,
)
_INFORMATIONAL_REQUEST_RE = re.compile(
    r"^\s*(?:what\b|why\b|who\b|when\b|where\b|how\s+(?:does|do|is|are|can|would|should)\b|"
    r"explain\b|describe\b|define\b|tell\s+me\s+about\b)",
    re.IGNORECASE,
)
_CREATIVE_REQUEST_RE = re.compile(
    r"^\s*(?:(?:write|create|make|compose)\s+(?:me\s+)?(?:(?:a|an|some)\s+)?"
    r"(?:poem|story|essay|joke|song|letter|email|description|caption|slogan|copy)\b"
    r"|play\s+devil['?]?s\s+advocate\b"
    r"|send\s+me\s+(?:(?:a|an|the)\s+)?(?:explanation|summary|overview|answer|idea|ideas|information)\b)",
    re.IGNORECASE,
)
_ACTION_REQUEST_RE = re.compile(
    r"\b(?:book|call|check|connect|control|copy|delete|download|email|enable|find|install|"
    r"launch|move|open|play|read|remove|schedule|search|send|set|start|stop|sync|text|turn|"
    r"upload)\b",
    re.IGNORECASE,
)
_CONVERSATIONAL_ACTION_RE = re.compile(
    r"^\s*(?:"
    r"find\s+(?:me\s+)?(?:(?:a|an|the)\s+)?(?:explanation|summary|overview|answer|idea|information)\b"
    r"|open\s+(?:(?:a|the)\s+)?(?:discussion|conversation|debate)\b"
    r"|connect\s+(?:the\s+)?(?:concept|idea|dots)\b"
    r"|(?:open|show|go\s+to|take\s+me\s+to)\b.{0,80}\b(?:settings|page|screen|tab)\b"
    r")",
    re.IGNORECASE,
)


class RecoveryActionKind(StrEnum):
    """User-visible recovery actions. None of these grant authority by themselves."""

    USE_CAPABILITY = "use_capability"
    ENABLE_CAPABILITY = "enable_capability"
    REQUEST_PERMISSION = "request_permission"
    IDENTIFY_USER = "identify_user"
    CONNECT_PROVIDER = "connect_provider"
    COMPOSE_CAPABILITIES = "compose_capabilities"
    BUILD_CAPABILITY = "build_capability"


@dataclass(frozen=True, slots=True)
class RecoveryAction:
    kind: RecoveryActionKind
    label: str
    detail: str
    source: str
    target: str | None = None
    targets: tuple[str, ...] = ()
    settings_route: str | None = None
    required_permissions: tuple[str, ...] = ()
    requires_confirmation: bool = False

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": self.kind.value,
            "label": self.label,
            "detail": self.detail,
            "source": self.source,
            "requires_confirmation": self.requires_confirmation,
        }
        if self.target:
            payload["target"] = self.target
        if self.targets:
            payload["targets"] = list(self.targets)
        if self.settings_route:
            payload["settings_route"] = self.settings_route
        if self.required_permissions:
            payload["required_permissions"] = list(self.required_permissions)
        return payload


@dataclass(frozen=True, slots=True)
class RecoveryPlan:
    message: str
    actions: tuple[RecoveryAction, ...] = ()
    searched_sources: tuple[str, ...] = ()
    blocked: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "message": self.message,
            "actions": [action.to_dict() for action in self.actions],
            "searched_sources": list(self.searched_sources),
            "blocked": self.blocked,
        }


@dataclass(frozen=True, slots=True)
class ExternalCapabilityCandidate:
    """Metadata supplied by an already configured external provider adapter."""

    id: str
    source: Literal["mcp", "openapi"]
    description: str
    triggers: tuple[str, ...] = ()
    enabled: bool = True
    required_permissions: tuple[str, ...] = ()
    risk: Literal["safe", "sensitive", "prohibited"] = "safe"
    operation: Literal["read", "mutation"] = "read"
    settings_route: str | None = None
    requires_identity: bool = False
    health: str = "unknown"


@dataclass(frozen=True, slots=True)
class _RankedCandidate:
    id: str
    source: str
    score: float
    enabled: bool
    required_permissions: tuple[str, ...]
    requires_config: tuple[str, ...]
    risk: str
    operation: str
    settings_route: str | None = None
    requires_identity: bool = False
    integration_state: str | None = None


class CapabilityGapResolver:
    """Search capability-gap sources in the approved, fail-closed order."""

    def __init__(
        self,
        registry: CapabilityRegistry,
        *,
        mcp_candidates: (
            list[ExternalCapabilityCandidate] | tuple[ExternalCapabilityCandidate, ...]
        ) = (),
        openapi_candidates: (
            list[ExternalCapabilityCandidate] | tuple[ExternalCapabilityCandidate, ...]
        ) = (),
        config: Any = None,
    ) -> None:
        self._registry = registry
        self._config = config
        self._mcp = tuple(mcp_candidates)
        self._openapi = tuple(openapi_candidates)

    def resolve(
        self,
        query: str,
        *,
        user_id: str | None,
        granted_permissions: set[str] | frozenset[str],
        allow_build: bool = True,
    ) -> RecoveryPlan:
        permissions = frozenset(granted_permissions)
        searched: list[str] = []

        stages: tuple[tuple[str, list[_RankedCandidate]], ...] = (
            ("local_enabled", self._local_candidates(query, enabled=True, source="local")),
            ("local_disabled", self._local_candidates(query, enabled=False, source="local")),
            ("openclaw", self._local_candidates(query, enabled=None, source="openclaw")),
            ("mcp", self._external_candidates(query, self._mcp)),
            ("openapi", self._external_candidates(query, self._openapi)),
        )

        partial_local: list[_RankedCandidate] = []
        for stage_name, candidates in stages:
            searched.append(stage_name)
            if stage_name == "local_enabled":
                partial_local = [c for c in candidates if 0 < c.score < _DIRECT_MATCH_THRESHOLD]
            direct = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.score >= _DIRECT_MATCH_THRESHOLD
                ),
                None,
            )
            if direct is None:
                moderate = [candidate for candidate in candidates if candidate.score >= 0.5]
                # A single moderate match is safer than skipping a clearly related
                # disabled/permissioned capability. Multiple moderate local matches
                # are deferred to the composition stage instead of guessed between.
                if len(moderate) == 1:
                    direct = moderate[0]
            if direct is None:
                continue
            plan = self._plan_for_candidate(direct, user_id, permissions, searched)
            if plan is not None:
                return plan

        searched.append("composition")
        composition = self._composition_plan(partial_local, user_id, permissions, searched)
        if composition is not None:
            return composition

        if not allow_build:
            return RecoveryPlan(message="", searched_sources=tuple(searched))

        searched.append("forge")
        return RecoveryPlan(
            message=(
                "I don't currently have an approved capability that can do that. "
                "I searched local capabilities and configured external capability sources first. "
                "I can help design a new bounded capability, but I need your approval before any build work."
            ),
            actions=(
                RecoveryAction(
                    kind=RecoveryActionKind.BUILD_CAPABILITY,
                    label="Approve capability build",
                    detail="Design and test a new bounded capability after explicit approval.",
                    source="forge",
                    requires_confirmation=True,
                ),
            ),
            searched_sources=tuple(searched),
        )

    def _local_candidates(
        self,
        query: str,
        *,
        enabled: bool | None,
        source: Literal["local", "openclaw"],
    ) -> list[_RankedCandidate]:
        candidates: list[_RankedCandidate] = []
        for card in self._registry.list(include_disabled=True):
            external_sources = {"openclaw", "mcp", "openapi"}
            if source == "local" and card.source in external_sources:
                continue
            if source == "openclaw" and card.source != "openclaw":
                continue
            if enabled is not None and card.enabled is not enabled:
                continue
            if card.health in _BLOCKED_HEALTH:
                continue
            score = _capability_score(query, card)
            if score <= 0:
                continue
            candidates.append(
                _RankedCandidate(
                    id=card.id,
                    source=card.source,
                    score=score,
                    enabled=card.enabled,
                    required_permissions=card.required_permissions,
                    requires_config=card.requires_config,
                    risk=card.risk,
                    operation=card.operation,
                    requires_identity=card.requires_identity,
                    integration_state=card.integration_state,
                )
            )
        return sorted(candidates, key=lambda candidate: (-candidate.score, candidate.id))

    @staticmethod
    def _external_candidates(
        query: str, candidates: tuple[ExternalCapabilityCandidate, ...]
    ) -> list[_RankedCandidate]:
        ranked: list[_RankedCandidate] = []
        for candidate in candidates:
            if candidate.health in _BLOCKED_HEALTH:
                continue
            text = " ".join(
                (candidate.id.replace("_", " "), candidate.description, *candidate.triggers)
            )
            score = _external_score(query, candidate, text)
            if score <= 0:
                continue
            ranked.append(
                _RankedCandidate(
                    id=candidate.id,
                    source=candidate.source,
                    score=score,
                    enabled=candidate.enabled,
                    required_permissions=candidate.required_permissions,
                    requires_config=(),
                    risk=candidate.risk,
                    operation=candidate.operation,
                    settings_route=candidate.settings_route,
                    requires_identity=candidate.requires_identity,
                )
            )
        return sorted(ranked, key=lambda candidate: (-candidate.score, candidate.id))

    def _plan_for_candidate(
        self,
        candidate: _RankedCandidate,
        user_id: str | None,
        permissions: frozenset[str],
        searched: list[str],
    ) -> RecoveryPlan | None:
        if candidate.risk == "prohibited":
            return RecoveryPlan(
                message="I found a matching capability, but it is prohibited by Rex policy and cannot be offered.",
                searched_sources=tuple(searched),
                blocked=True,
            )

        if candidate.requires_identity and not user_id:
            return RecoveryPlan(
                message="I found a matching capability, but it requires an identified Rex user.",
                actions=(
                    RecoveryAction(
                        kind=RecoveryActionKind.IDENTIFY_USER,
                        label="Identify the current user",
                        detail=(
                            "Sign in or identify the current Rex profile, then retry this "
                            "identity-scoped request."
                        ),
                        source=candidate.source,
                        target=candidate.id,
                    ),
                ),
                searched_sources=tuple(searched),
            )

        missing = (
            ()
            if "admin" in permissions
            else tuple(sorted(set(candidate.required_permissions) - set(permissions)))
        )
        if missing:
            return RecoveryPlan(
                message="I found a matching capability, but your current permissions do not authorize it.",
                actions=(
                    RecoveryAction(
                        kind=RecoveryActionKind.REQUEST_PERMISSION,
                        label="Request access",
                        detail=(
                            f"Ask an administrator to grant {', '.join(missing)} to your Rex profile, "
                            "then retry the request."
                        ),
                        source=candidate.source,
                        target=candidate.id,
                        required_permissions=missing,
                        requires_confirmation=True,
                    ),
                ),
                searched_sources=tuple(searched),
            )

        missing_config = tuple(
            key for key in candidate.requires_config if not self._config_value(key)
        )
        if missing_config:
            return RecoveryPlan(
                message="I found a matching capability, but required configuration is missing.",
                actions=(
                    RecoveryAction(
                        kind=RecoveryActionKind.ENABLE_CAPABILITY,
                        label=f"Configure {candidate.id}",
                        detail=(
                            f"Required Rex config key(s): {', '.join(missing_config)}. "
                            "Configure them through the existing settings/credential source, "
                            "then retry."
                        ),
                        source=candidate.source,
                        target=candidate.id,
                        settings_route=candidate.settings_route,
                        requires_confirmation=True,
                    ),
                ),
                searched_sources=tuple(searched),
            )

        if candidate.integration_state == "unconfigured":
            return RecoveryPlan(
                message="I found a matching integration, but its integration state is unconfigured.",
                actions=(
                    RecoveryAction(
                        kind=RecoveryActionKind.ENABLE_CAPABILITY,
                        label=f"Configure {candidate.id}",
                        detail=(
                            "Complete the integration configuration required by this capability, "
                            "then retry the request."
                        ),
                        source=candidate.source,
                        target=candidate.id,
                        settings_route=candidate.settings_route,
                        requires_confirmation=True,
                    ),
                ),
                searched_sources=tuple(searched),
            )

        if candidate.integration_state == "unavailable":
            return RecoveryPlan(
                message=(
                    "I found a matching integration, but it is unavailable in the current "
                    "runtime. No supported configuration path is currently advertised."
                ),
                searched_sources=tuple(searched),
                blocked=True,
            )

        if candidate.source in {"openclaw", "mcp", "openapi"}:
            return RecoveryPlan(
                message=f"I found a matching capability through {candidate.source}, but it is not wired into this Rex execution path yet.",
                actions=(
                    RecoveryAction(
                        kind=RecoveryActionKind.CONNECT_PROVIDER,
                        label=f"Connect {candidate.id}",
                        detail="Open the configured external capability flow and review permissions before enabling it.",
                        source=candidate.source,
                        target=candidate.id,
                        settings_route=candidate.settings_route,
                        requires_confirmation=True,
                    ),
                ),
                searched_sources=tuple(searched),
            )

        if not candidate.enabled:
            return RecoveryPlan(
                message="I found a matching capability, but it is not currently enabled.",
                actions=(
                    RecoveryAction(
                        kind=RecoveryActionKind.ENABLE_CAPABILITY,
                        label=f"Enable {candidate.id}",
                        detail="Enable this capability before Rex can use it, then retry.",
                        source=candidate.source,
                        target=candidate.id,
                        settings_route=candidate.settings_route,
                        requires_confirmation=True,
                    ),
                ),
                searched_sources=tuple(searched),
            )

        return RecoveryPlan(
            message="I found an approved capability that can handle this request.",
            actions=(
                RecoveryAction(
                    kind=RecoveryActionKind.USE_CAPABILITY,
                    label=f"Use {candidate.id}",
                    detail="Use the already enabled and authorized capability.",
                    source=candidate.source,
                    target=candidate.id,
                ),
            ),
            searched_sources=tuple(searched),
        )

    def _config_value(self, key: str) -> object:
        if self._config is None:
            return None
        if isinstance(self._config, dict):
            return self._config.get(key)
        return getattr(self._config, key, None)

    def _composition_plan(
        self,
        candidates: list[_RankedCandidate],
        user_id: str | None,
        permissions: frozenset[str],
        searched: list[str],
    ) -> RecoveryPlan | None:
        safe = [
            candidate
            for candidate in candidates
            if candidate.risk == "safe"
            and candidate.operation == "read"
            and (not candidate.requires_identity or bool(user_id))
            and candidate.integration_state not in {"unconfigured", "unavailable"}
            and not any(not self._config_value(key) for key in candidate.requires_config)
            and (
                "admin" in permissions or set(candidate.required_permissions).issubset(permissions)
            )
        ]
        if len(safe) < 2:
            return None
        targets = tuple(candidate.id for candidate in safe[:3])
        return RecoveryPlan(
            message="I can satisfy this with a safe declarative composition of existing read-only capabilities.",
            actions=(
                RecoveryAction(
                    kind=RecoveryActionKind.COMPOSE_CAPABILITIES,
                    label="Compose existing capabilities",
                    detail="Combine already enabled, authorized, read-only capabilities without generated code.",
                    source="composition",
                    targets=targets,
                    requires_confirmation=True,
                ),
            ),
            searched_sources=tuple(searched),
        )


def _tokens(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in _TOKEN_RE.findall(text.casefold().replace("_", " "))
        if token not in _STOPWORDS
    )


def _score(query: str, candidate_text: str) -> float:
    query_tokens = _tokens(query)
    candidate_tokens = _tokens(candidate_text)
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap_tokens = query_tokens & candidate_tokens
    if not (overlap_tokens - _GENERIC_ACTION_TOKENS):
        return 0.0
    return len(overlap_tokens) / min(len(query_tokens), len(candidate_tokens))


def _capability_score(query: str, card: Capability) -> float:
    identity_texts = (card.id.replace("_", " "), *card.triggers, *card.examples)
    identity_score = max((_score(query, text) for text in identity_texts if text), default=0.0)
    descriptive_score = _score(query, f"{card.description} {card.category}") * 0.9
    return max(identity_score, descriptive_score)


def _external_score(
    query: str, candidate: ExternalCapabilityCandidate, candidate_text: str
) -> float:
    identity_texts = (candidate.id.replace("_", " "), *candidate.triggers)
    identity_score = max((_score(query, text) for text in identity_texts if text), default=0.0)
    descriptive_score = _score(query, candidate_text) * 0.9
    return max(identity_score, descriptive_score)


def looks_like_action_request(text: str) -> bool:
    """Return whether text is asking Rex to perform an external/tool action."""
    stripped = text.strip()
    if not stripped:
        return False
    if (
        _INFORMATIONAL_REQUEST_RE.search(stripped)
        or _CREATIVE_REQUEST_RE.search(stripped)
        or _CONVERSATIONAL_ACTION_RE.search(stripped)
    ):
        return False
    return bool(_ACTION_REQUEST_RE.search(stripped) or _CAPABILITY_BUILD_RE.search(stripped))


def looks_like_capability_request(text: str) -> bool:
    """Return whether text explicitly requests creating/building a capability."""
    return bool(_CAPABILITY_BUILD_RE.search(text))


__all__ = [
    "CapabilityGapResolver",
    "ExternalCapabilityCandidate",
    "RecoveryAction",
    "RecoveryActionKind",
    "RecoveryPlan",
    "looks_like_action_request",
    "looks_like_capability_request",
]
