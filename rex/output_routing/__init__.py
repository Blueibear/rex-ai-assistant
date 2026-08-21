"""Per-user output-routing policy and resolution."""

from .audit import install_output_routing_audit
from .models import (
    FallbackMode,
    OutputKind,
    QuietHours,
    ResolvedRoute,
    RoutingRule,
    UserOutputPolicy,
)
from .service import OutputRoutingService

# Installing here keeps direct ``rex.output_routing.service`` imports and
# package imports on the same audited resolver without changing policy logic.
install_output_routing_audit(OutputRoutingService)

__all__ = [
    "FallbackMode",
    "OutputKind",
    "OutputRoutingService",
    "QuietHours",
    "ResolvedRoute",
    "RoutingRule",
    "UserOutputPolicy",
]
