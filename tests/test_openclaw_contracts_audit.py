"""Audit for rex/contracts/ migration interfaces (US-P7-017).

Verifies that:
  - All contracts are importable and structurally valid
  - Each contract still has a corresponding legacy module (not yet retired)
  - Contracts won't be cleaned up until their legacy modules are retired

Status:
  - plugin_loader.py retired → plugins.py contract removed (Phase 7)
  - browser_automation.py retired → browser.py contract removed (Phase 7 / iter 81)
  - tool_router.py retired (US-P7-008) → tool_routing.py contract now points to
    rex/openclaw/tool_executor.py
  - event_bus.py retired (US-P7-002) → event_bus.py contract removed (iter 96)
  Remaining: none — all OPENCLAW-REPLACE modules retired.
"""

from __future__ import annotations

import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
REX_PKG = REPO_ROOT / "rex"
CONTRACTS_PKG = REX_PKG / "contracts"

# Map: contract file → legacy module it replaces
# Note: plugins.py contract was removed when rex/plugin_loader.py was retired.
# Note: browser.py contract was removed when rex/browser_automation.py was retired.
# Note: event_bus.py contract was removed when rex/event_bus.py was retired (iter 96).
CONTRACT_TO_LEGACY = {
    "tool_routing.py": "rex/openclaw/tool_executor.py",
}

# Contracts that are always kept (not tied to a single retiring module)
# Note: dashboard.py contract was removed when rex/dashboard/ was retired (iter 93)
PERMANENT_CONTRACTS = {"core.py", "version.py"}


class TestContractsAudit:
    """Verify contracts/ directory health."""

    def test_all_contracts_exist(self):
        """All expected contract files are present."""
        for fname in CONTRACT_TO_LEGACY:
            assert (CONTRACTS_PKG / fname).exists(), f"Missing contract: {fname}"
        for fname in PERMANENT_CONTRACTS:
            assert (CONTRACTS_PKG / fname).exists(), f"Missing permanent contract: {fname}"

    def test_contracts_are_importable(self):
        """All contracts can be imported without errors."""
        import importlib

        for fname in list(CONTRACT_TO_LEGACY) + list(PERMANENT_CONTRACTS):
            module_name = "rex.contracts." + fname.removesuffix(".py")
            try:
                importlib.import_module(module_name)
            except ImportError as exc:
                raise AssertionError(f"Contract {module_name} failed to import: {exc}") from exc

    def test_legacy_modules_still_exist_for_each_contract(self):
        """Each contract's legacy module still exists — contracts are still needed."""
        for contract_file, legacy_path in CONTRACT_TO_LEGACY.items():
            legacy = REPO_ROOT / legacy_path
            assert legacy.exists(), (
                f"Legacy module {legacy_path} was removed but contract {contract_file} still exists.\n"
                "Remove the contract when its legacy module is retired."
            )

    def test_no_orphaned_contracts(self):
        """No contract files exist for already-retired modules."""
        for contract_file, legacy_path in CONTRACT_TO_LEGACY.items():
            legacy = REPO_ROOT / legacy_path
            contract = CONTRACTS_PKG / contract_file
            if not legacy.exists() and contract.exists():
                raise AssertionError(
                    f"Orphaned contract: {contract_file} — legacy module {legacy_path} "
                    "has been retired but the contract was not cleaned up."
                )

    def test_contracts_cleanup_blocked(self):
        """All legacy modules still present — contracts cleanup is not yet due."""
        remaining = [
            legacy for legacy in CONTRACT_TO_LEGACY.values() if (REPO_ROOT / legacy).exists()
        ]
        assert len(remaining) == len(CONTRACT_TO_LEGACY), (
            f"Some legacy modules retired: {set(CONTRACT_TO_LEGACY.values()) - set(remaining)}\n"
            "Clean up corresponding contracts now."
        )
