#!/usr/bin/env python3
"""Enforce the documented pytest skipped-test budget (US-037).

The CI test job captures pytest output with ``-rs`` and passes the report to
this script. The gate fails closed when the report cannot be parsed and fails
when the executed skip count exceeds the approved baseline.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

SKIP_BUDGET = 82

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_COUNTER_RE = re.compile(
    r"(?P<count>\d+)\s+"
    r"(?P<label>passed|failed|errors?|skipped|deselected|warnings?|xfailed|xpassed)\b"
)
_DURATION_RE = re.compile(r"\bin\s+\d+(?:\.\d+)?s(?:\s|$)")
_EXECUTION_LABELS = {"passed", "failed", "error", "errors"}


def parse_pytest_summary(output: str) -> dict[str, int]:
    """Return counters from the final pytest execution summary in *output*.

    A valid summary must contain a duration and at least one execution counter.
    This avoids mistaking ``SKIPPED [1]`` reason lines or unrelated prose for
    the terminal summary.
    """
    for raw_line in reversed(output.splitlines()):
        line = _ANSI_ESCAPE_RE.sub("", raw_line).strip(" =")
        if not _DURATION_RE.search(line):
            continue
        counters = {
            match.group("label"): int(match.group("count")) for match in _COUNTER_RE.finditer(line)
        }
        if counters.keys() & _EXECUTION_LABELS:
            return counters
    raise ValueError("pytest execution summary was not found")


def count_skipped_tests(output: str) -> int:
    """Return the skipped-test count from the final pytest summary."""
    return parse_pytest_summary(output).get("skipped", 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Path to captured pytest output")
    parser.add_argument(
        "--budget",
        type=int,
        default=SKIP_BUDGET,
        help=f"Maximum allowed skipped tests (default: {SKIP_BUDGET})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.budget < 0:
        print("ERROR: skip budget must be zero or greater.", file=sys.stderr)
        return 2

    try:
        output = args.report.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"ERROR: unable to read pytest report {args.report}: {exc}", file=sys.stderr)
        return 2

    try:
        skipped = count_skipped_tests(output)
    except ValueError as exc:
        print(f"ERROR: unable to parse {args.report}: {exc}", file=sys.stderr)
        return 2

    if skipped > args.budget:
        excess = skipped - args.budget
        print(
            f"ERROR: pytest skipped {skipped} tests, exceeding the budget "
            f"of {args.budget} by {excess}."
        )
        print(
            "Reduce or repair skipped tests. Do not raise the budget without "
            "updating the documented inventory and rationale."
        )
        return 1

    print(f"OK: pytest skipped {skipped} tests; budget is {args.budget}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
