"""Write .env files safely with backups and template preservation."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .env_schema import EnvSchema, parse_env_example


def create_backup(env_path: Path, backup_dir: Optional[Path] = None) -> Optional[Path]:
    """Create a timestamped backup of .env file.

    Args:
        env_path: Path to .env file to backup
        backup_dir: Directory for backups (default: <repo_root>/backups)

    Returns:
        Path to backup file, or None if env_path doesn't exist
    """
    if not env_path.exists():
        return None

    if backup_dir is None:
        backup_dir = env_path.parent / "backups"

    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f".env.backup.{timestamp}"

    shutil.copy2(env_path, backup_path)
    return backup_path


def read_current_env(env_path: Path) -> Dict[str, str]:
    """Read current .env file into a dictionary.

    Args:
        env_path: Path to .env file

    Returns:
        Dictionary of KEY=VALUE pairs
    """
    if not env_path.exists():
        return {}

    env_vars = {}
    content = env_path.read_text(encoding='utf-8')

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        if '=' in line:
            key, value = line.split('=', 1)
            env_vars[key.strip()] = value.strip()

    return env_vars


def write_env_from_template(
    env_path: Path,
    template_path: Path,
    values: Dict[str, str],
    custom_overrides: Optional[Dict[str, str]] = None,
    create_backup: bool = True,
) -> Path:
    """Write .env file using template structure with updated values.

    This preserves the structure, sections, and comments from .env.example
    while substituting the actual values.

    Args:
        env_path: Path to write .env file
        template_path: Path to .env.example template
        values: Dictionary of KEY=VALUE pairs to write
        custom_overrides: Additional custom keys not in template
        create_backup: Whether to backup existing .env first

    Returns:
        Path to backup file if created, otherwise env_path
    """
    # Create backup first
    backup_path = None
    if create_backup and env_path.exists():
        backup_path = globals()['create_backup'](env_path)

    # Read template
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    template_content = template_path.read_text(encoding='utf-8')
    lines = template_content.splitlines()

    output_lines = []
    for line in lines:
        # Preserve comments and empty lines
        if not line.strip() or line.strip().startswith('#'):
            output_lines.append(line)
            continue

        # Key=Value line
        if '=' in line:
            key = line.split('=', 1)[0].strip()
            if key in values:
                output_lines.append(f"{key}={values[key]}")
            else:
                output_lines.append(line)  # Keep default
        else:
            output_lines.append(line)

    # Add custom overrides section if any
    if custom_overrides:
        output_lines.append('')
        output_lines.append('# ================================')
        output_lines.append('# Custom Overrides')
        output_lines.append('# ================================')
        output_lines.append('# Additional settings not in .env.example')
        output_lines.append('')

        for key, value in sorted(custom_overrides.items()):
            output_lines.append(f"{key}={value}")

    # Write to file
    output_content = '\n'.join(output_lines) + '\n'
    env_path.write_text(output_content, encoding='utf-8')

    return backup_path or env_path


def restore_from_backup(backup_path: Path, env_path: Path) -> None:
    """Restore .env from a backup file.

    Args:
        backup_path: Path to backup file
        env_path: Path to .env file to restore
    """
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup not found: {backup_path}")

    shutil.copy2(backup_path, env_path)


def get_backup_files(backup_dir: Path) -> list[Path]:
    """Get list of backup files sorted by timestamp (newest first).

    Args:
        backup_dir: Directory containing backups

    Returns:
        List of backup file paths
    """
    if not backup_dir.exists():
        return []

    backups = list(backup_dir.glob(".env.backup.*"))
    return sorted(backups, reverse=True)


def get_extra_keys(env_path: Path, schema: EnvSchema) -> Dict[str, str]:
    """Find keys in .env that are not in .env.example.

    Args:
        env_path: Path to .env file
        schema: Parsed schema from .env.example

    Returns:
        Dictionary of extra KEY=VALUE pairs
    """
    if not env_path.exists():
        return {}

    current_env = read_current_env(env_path)
    schema_keys = {var.key for var in schema.get_all_variables()}

    extra_keys = {}
    for key, value in current_env.items():
        if key not in schema_keys:
            extra_keys[key] = value

    return extra_keys
