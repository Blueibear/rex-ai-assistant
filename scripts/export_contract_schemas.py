#!/usr/bin/env python3
"""Export Rex contract schemas to JSON Schema files.

This script generates JSON Schema documentation for all Rex contract models,
enabling external tools and components to validate data against these schemas.

Usage:
    python scripts/export_contract_schemas.py

Output:
    docs/contracts/
        index.json              - Index of all schemas with metadata
        EvidenceRef.json        - Schema for evidence references
        ToolCall.json           - Schema for tool call requests
        ToolResult.json         - Schema for tool results
        Approval.json           - Schema for approval requests
        Action.json             - Schema for actions
        Task.json               - Schema for tasks
        Notification.json       - Schema for notifications

Environment Variables:
    REX_CONTRACTS_OUTPUT_DIR   - Override the output directory (default: docs/contracts)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_output_dir(project_root: Path) -> Path:
    """Get the output directory for schema files."""
    env_dir = os.environ.get("REX_CONTRACTS_OUTPUT_DIR")
    if env_dir:
        return Path(env_dir)
    return project_root / "docs" / "contracts"


def export_schemas(output_dir: Path | None = None, project_root: Path | None = None) -> dict[str, str]:
    """Export all contract schemas to JSON files.

    Args:
        output_dir: Directory to write schema files. If None, uses default.
        project_root: Project root directory.

    Returns:
        Dictionary mapping model names to their output file paths.
    """
    from rex.contracts import CONTRACT_VERSION
    from rex.contracts.core import ALL_MODELS

    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    if output_dir is None:
        output_dir = get_output_dir(project_root)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_files: dict[str, str] = {}
    schema_entries: list[dict[str, str]] = []

    for model in ALL_MODELS:
        model_name = model.__name__
        schema = model.model_json_schema()

        # Add contract version to schema metadata
        schema["$comment"] = f"Rex Contract Schema v{CONTRACT_VERSION}"

        schema_filename = f"{model_name}.json"
        schema_path = output_dir / schema_filename

        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
            f.write("\n")

        exported_files[model_name] = str(schema_path)
        schema_entries.append({
            "model": model_name,
            "file": schema_filename,
            "description": model.__doc__.split("\n")[0] if model.__doc__ else "",
        })

        print(f"  Exported: {schema_filename}")

    # Generate index file
    index = {
        "contract_version": CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schemas": schema_entries,
    }

    index_path = output_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"  Exported: index.json")
    exported_files["index"] = str(index_path)

    return exported_files


def main() -> int:
    """Main entry point for the export script."""
    # Ensure the project root is in the path
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from rex.contracts import CONTRACT_VERSION

    output_dir = get_output_dir(project_root)
    print(f"Exporting Rex contract schemas v{CONTRACT_VERSION}")
    print(f"Output directory: {output_dir}")
    print()

    try:
        exported = export_schemas(output_dir, project_root)
        print()
        print(f"Successfully exported {len(exported)} files.")
        return 0
    except Exception as e:
        print(f"Error exporting schemas: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
