"""
VS Code integration service.

Provides VS Code-like file operations and testing capabilities:
- File reading and display
- Patch/diff application
- Test execution
- Policy-gated operations
- Audit logging
"""

import difflib
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rex.audit import LogEntry, get_audit_logger
from rex.contracts.core import ToolCall
from rex.policy_engine import get_policy_engine


@dataclass
class PatchResult:
    """Result of applying a patch."""

    success: bool
    file_path: str
    hunks_applied: int
    hunks_failed: int
    errors: list[str]


@dataclass
class PyTestRunResult:
    """Result of running tests."""

    success: bool
    total: int
    passed: int
    failed: int
    errors: int
    skipped: int
    duration_seconds: float
    output: str


class VSCodeService:
    """
    Service for VS Code-like operations.

    Features:
    - File reading with syntax-aware display
    - Unified diff patch application
    - Test execution with pytest
    - Policy checks for file modifications
    - Audit logging for all operations
    """

    def __init__(self, workspace_path: str | None = None):
        """
        Initialize VS Code service.

        Args:
            workspace_path: Base workspace directory (default: current directory)
        """
        self.workspace_path = Path(workspace_path or ".")
        self._audit_logger = get_audit_logger()
        self._policy_engine = get_policy_engine()

    def open_file(self, file_path: str) -> dict[str, Any]:
        """
        Open and read a file.

        Args:
            file_path: Path to file (relative to workspace)

        Returns:
            Dictionary with file content and metadata
        """
        action_id = str(uuid.uuid4())
        start_time = datetime.now()

        try:
            full_path = (self.workspace_path / file_path).resolve()

            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Read file content
            with open(full_path, encoding="utf-8") as f:
                content = f.read()

            # Get file metadata
            stat = full_path.stat()

            result = {
                "status": "success",
                "path": str(full_path),
                "content": content,
                "lines": len(content.splitlines()),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Audit log
            self._audit_logger.log(
                LogEntry(
                    action_id=action_id,
                    tool="code_open_file",
                    tool_call_args={"file_path": file_path},
                    policy_decision="allowed",
                    tool_result={"path": str(full_path), "lines": result["lines"]},
                    error=None,
                    duration_ms=duration_ms,
                )
            )

            return result

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            error_msg = f"Failed to open file: {str(e)}"

            self._audit_logger.log(
                LogEntry(
                    action_id=action_id,
                    tool="code_open_file",
                    tool_call_args={"file_path": file_path},
                    policy_decision="allowed",
                    tool_result=None,
                    error=error_msg,
                    duration_ms=duration_ms,
                )
            )

            raise RuntimeError(error_msg) from e

    def apply_patch(
        self,
        file_path: str,
        patch_content: str,
    ) -> PatchResult:
        """
        Apply a unified diff patch to a file.

        Args:
            file_path: Path to file to patch
            patch_content: Unified diff content

        Returns:
            PatchResult with application status
        """
        # Policy check for file modification
        tool_call = ToolCall(
            tool="code_apply_patch",
            args={"file_path": file_path},
            requested_by="user",
            created_at=datetime.now(),
        )
        decision = self._policy_engine.decide(tool_call, metadata={})

        if not decision.allowed:
            raise PermissionError(f"Policy denied patch application: {decision.reason}")

        action_id = str(uuid.uuid4())
        start_time = datetime.now()

        try:
            full_path = (self.workspace_path / file_path).resolve()

            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Read current file content
            with open(full_path, encoding="utf-8") as f:
                original_lines = f.readlines()

            # Parse and apply patch
            patched_lines, hunks_applied, hunks_failed, errors = self._apply_unified_diff(
                original_lines,
                patch_content,
            )

            # Write patched content
            if hunks_failed == 0:
                with open(full_path, "w", encoding="utf-8") as f:
                    f.writelines(patched_lines)

            result = PatchResult(
                success=hunks_failed == 0,
                file_path=str(full_path),
                hunks_applied=hunks_applied,
                hunks_failed=hunks_failed,
                errors=errors,
            )

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Audit log
            self._audit_logger.log(
                LogEntry(
                    action_id=action_id,
                    tool="code_apply_patch",
                    tool_call_args={"file_path": file_path},
                    policy_decision="allowed",
                    tool_result={
                        "success": result.success,
                        "hunks_applied": hunks_applied,
                        "hunks_failed": hunks_failed,
                    },
                    error=None if result.success else f"{hunks_failed} hunks failed",
                    duration_ms=duration_ms,
                )
            )

            return result

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            error_msg = f"Patch application failed: {str(e)}"

            self._audit_logger.log(
                LogEntry(
                    action_id=action_id,
                    tool="code_apply_patch",
                    tool_call_args={"file_path": file_path},
                    policy_decision="allowed",
                    tool_result=None,
                    error=error_msg,
                    duration_ms=duration_ms,
                )
            )

            raise RuntimeError(error_msg) from e

    def _apply_unified_diff(
        self,
        original_lines: list[str],
        patch_content: str,
    ) -> tuple[list[str], int, int, list[str]]:
        """
        Apply a unified diff to file lines.

        Args:
            original_lines: Original file lines
            patch_content: Unified diff content

        Returns:
            Tuple of (patched_lines, hunks_applied, hunks_failed, errors)
        """
        # Simple patch application
        # In production, use a proper patch library like 'unidiff'
        # For now, we'll use a basic implementation

        result_lines = original_lines.copy()
        hunks_applied = 0
        hunks_failed = 0
        errors = []

        try:
            # Parse diff hunks
            lines = patch_content.splitlines()
            i = 0

            while i < len(lines):
                line = lines[i]

                # Look for hunk header: @@ -start,count +start,count @@
                if line.startswith("@@"):
                    match = re.match(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@", line)
                    if match:
                        old_start = int(match.group(1))
                        int(match.group(3))

                        # Collect hunk lines
                        hunk_lines = []
                        i += 1
                        while i < len(lines) and not lines[i].startswith("@@"):
                            if lines[i].startswith("---") or lines[i].startswith("+++"):
                                i += 1
                                continue
                            hunk_lines.append(lines[i])
                            i += 1

                        # Try to apply hunk
                        try:
                            result_lines = self._apply_hunk(
                                result_lines,
                                hunk_lines,
                                old_start - 1,  # Convert to 0-indexed
                            )
                            hunks_applied += 1
                        except Exception as e:
                            hunks_failed += 1
                            errors.append(f"Hunk at line {old_start} failed: {str(e)}")
                        continue

                i += 1

        except Exception as e:
            errors.append(f"Patch parsing failed: {str(e)}")
            hunks_failed += 1

        return result_lines, hunks_applied, hunks_failed, errors

    def _apply_hunk(
        self,
        lines: list[str],
        hunk_lines: list[str],
        start_line: int,
    ) -> list[str]:
        """Apply a single hunk to file lines."""
        result = lines[:start_line]
        line_idx = start_line

        for hunk_line in hunk_lines:
            if not hunk_line:
                continue

            prefix = hunk_line[0]
            content = hunk_line[1:] + "\n" if len(hunk_line) > 1 else "\n"

            if prefix == " ":  # Context line
                if line_idx < len(lines):
                    result.append(lines[line_idx])
                    line_idx += 1
            elif prefix == "-":  # Remove line
                if line_idx < len(lines):
                    line_idx += 1
            elif prefix == "+":  # Add line
                result.append(content)

        # Append remaining lines
        result.extend(lines[line_idx:])
        return result

    def run_tests(
        self,
        test_path: str | None = None,
        pattern: str | None = None,
        verbose: bool = False,
    ) -> PyTestRunResult:
        """
        Run tests using pytest.

        Args:
            test_path: Path to test file or directory (default: "tests/")
            pattern: Test pattern filter (e.g., "test_browser")
            verbose: Verbose output

        Returns:
            PyTestRunResult with test execution results
        """
        action_id = str(uuid.uuid4())
        start_time = datetime.now()

        try:
            # Build pytest command
            cmd = ["pytest"]

            if test_path:
                cmd.append(str(self.workspace_path / test_path))
            else:
                cmd.append("tests/")

            if pattern:
                cmd.extend(["-k", pattern])

            if verbose:
                cmd.append("-v")
            else:
                cmd.append("-q")

            # Add additional options
            cmd.extend(
                [
                    "--tb=short",
                    "--no-header",
                ]
            )

            # Run tests
            result = subprocess.run(
                cmd,
                cwd=str(self.workspace_path),
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            # Parse output
            output = result.stdout + result.stderr
            test_result = self._parse_pytest_output(output, result.returncode)

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Audit log
            self._audit_logger.log(
                LogEntry(
                    action_id=action_id,
                    tool="code_run_tests",
                    tool_call_args={"test_path": test_path, "pattern": pattern},
                    policy_decision="allowed",
                    tool_result={
                        "success": test_result.success,
                        "passed": test_result.passed,
                        "failed": test_result.failed,
                    },
                    error=None if test_result.success else f"{test_result.failed} tests failed",
                    duration_ms=duration_ms,
                )
            )

            return test_result

        except subprocess.TimeoutExpired:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            error_msg = "Tests timed out after 5 minutes"

            self._audit_logger.log(
                LogEntry(
                    action_id=action_id,
                    tool="code_run_tests",
                    tool_call_args={"test_path": test_path},
                    policy_decision="allowed",
                    tool_result=None,
                    error=error_msg,
                    duration_ms=duration_ms,
                )
            )

            return PyTestRunResult(
                success=False,
                total=0,
                passed=0,
                failed=0,
                errors=1,
                skipped=0,
                duration_seconds=300,
                output=error_msg,
            )

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            error_msg = f"Test execution failed: {str(e)}"

            self._audit_logger.log(
                LogEntry(
                    action_id=action_id,
                    tool="code_run_tests",
                    tool_call_args={"test_path": test_path},
                    policy_decision="allowed",
                    tool_result=None,
                    error=error_msg,
                    duration_ms=duration_ms,
                )
            )

            raise RuntimeError(error_msg) from e

    def _parse_pytest_output(self, output: str, returncode: int) -> PyTestRunResult:
        """
        Parse pytest output to extract test results.

        Args:
            output: Pytest output text
            returncode: Process return code

        Returns:
            PyTestRunResult with parsed data
        """
        # Default values
        passed = 0
        failed = 0
        errors = 0
        skipped = 0
        duration = 0.0

        # Parse summary line
        # Example: "5 passed, 2 failed in 1.23s"
        summary_pattern = r"(\d+)\s+passed"
        match = re.search(summary_pattern, output)
        if match:
            passed = int(match.group(1))

        failed_pattern = r"(\d+)\s+failed"
        match = re.search(failed_pattern, output)
        if match:
            failed = int(match.group(1))

        error_pattern = r"(\d+)\s+error"
        match = re.search(error_pattern, output)
        if match:
            errors = int(match.group(1))

        skipped_pattern = r"(\d+)\s+skipped"
        match = re.search(skipped_pattern, output)
        if match:
            skipped = int(match.group(1))

        duration_pattern = r"in\s+([\d.]+)s"
        match = re.search(duration_pattern, output)
        if match:
            duration = float(match.group(1))

        total = passed + failed + errors + skipped

        return PyTestRunResult(
            success=returncode == 0,
            total=total,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            duration_seconds=duration,
            output=output,
        )

    def generate_diff(self, file_path: str, new_content: str) -> str:
        """
        Generate a unified diff for proposed changes.

        Args:
            file_path: Path to file
            new_content: Proposed new content

        Returns:
            Unified diff string
        """
        full_path = (self.workspace_path / file_path).resolve()

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(full_path, encoding="utf-8") as f:
            original_content = f.read()

        original_lines = original_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )

        return "".join(diff)

    def list_files(
        self,
        directory: str = ".",
        pattern: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List files in a directory.

        Args:
            directory: Directory to list (relative to workspace)
            pattern: Optional glob pattern (e.g., "*.py")

        Returns:
            List of file information dictionaries
        """
        workspace_root = Path(self.workspace_path).resolve()
        dir_path = (workspace_root / directory).resolve()

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        files = []

        if pattern:
            matched_files = dir_path.glob(pattern)
        else:
            matched_files = dir_path.iterdir()

        for file_path in matched_files:
            if file_path.is_file():
                stat = file_path.stat()
                resolved_file = file_path.resolve()
                try:
                    relative_path = str(resolved_file.relative_to(workspace_root))
                except ValueError:
                    relative_path = os.path.relpath(str(resolved_file), str(workspace_root))
                files.append(
                    {
                        "name": file_path.name,
                        "path": relative_path,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    }
                )

        return sorted(files, key=lambda x: x["name"])


# Singleton instance
_vscode_service: VSCodeService | None = None


def get_vscode_service() -> VSCodeService:
    """Get or create the global VS Code service."""
    global _vscode_service
    if _vscode_service is None:
        _vscode_service = VSCodeService()
    return _vscode_service


def set_vscode_service(service: VSCodeService) -> None:
    """Set the global VS Code service."""
    global _vscode_service
    _vscode_service = service


def reset_vscode_service() -> None:
    """Reset the global VS Code service (for testing)."""
    global _vscode_service
    _vscode_service = None
