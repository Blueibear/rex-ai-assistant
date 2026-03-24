"""Contract versioning for Rex schemas.

This module defines the contract version that all Rex components should use
to ensure compatibility when communicating via these schemas.

Versioning Policy:
    - Major: Breaking changes (removed/renamed fields, changed types)
    - Minor: Backward-compatible additions (new optional fields)
    - Patch: Bug fixes, documentation updates

Example:
    from rex.contracts.version import CONTRACT_VERSION, get_version_info

    info = get_version_info()
    print(f"Using contracts v{info['version']}")
"""

from __future__ import annotations

CONTRACT_VERSION = "0.1.0"
"""Current contract schema version.

All schemas in rex.contracts are versioned together. Components should
check this version when deserializing messages from other components.
"""


def get_version_info() -> dict[str, str]:
    """Return version information and compatibility notes.

    Returns:
        A dictionary containing:
            - version: The current contract version string
            - compatibility: A note about backward compatibility
            - status: The stability status of these contracts

    Example:
        >>> info = get_version_info()
        >>> info['version']
        '0.1.0'
    """
    return {
        "version": CONTRACT_VERSION,
        "compatibility": (
            "Schemas are in initial development. Breaking changes may occur "
            "until version 1.0.0 is released."
        ),
        "status": "alpha",
    }
