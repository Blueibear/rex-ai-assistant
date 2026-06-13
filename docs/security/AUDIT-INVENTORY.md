# Security Audit Inventory

**Generated:** 2026-06-13
**Command:** `python scripts/security_audit.py`
**Exit code:** 0 (no secrets found; placeholder findings are classified below)
**Files scanned:** 1138 | **Files excluded:** 155

---

## Raw Audit Summary

```
Files scanned:  1138
Files excluded: 155  (cache dirs, egg-info, untracked, self-excluded)

MERGE CONFLICT MARKERS
CLEAN - No merge markers found

PLACEHOLDER/INCOMPLETE CODE MARKERS
Found 82 actionable findings (78 source-code, 4 configuration)
Plus 132 informational findings (132 documentation, 0 needs-review)

EXPOSED SECRETS
CLEAN - No exposed secrets found
```

---

## Triage Inventory

Classifications used:

- **production-blocker** — stub/placeholder on a reachable production path; must be fixed before release
- **dev-only-documented** — archived, developer-tool, or explicitly scoped non-production code; no user-facing impact
- **false-positive** — pattern matches legitimate non-stub usage (HTML attribute, sentinel character, test fixture, variable name, truncation feature)

### Production Blockers

| File | Lines | Marker / Excerpt | Classification | Resolution |
|------|-------|-----------------|----------------|-----------|
| `rex/openclaw/workflow_bridge.py` | 169 | `# TODO: replace with real OpenClaw workflow executor registration once API is confirmed` | **production-blocker** | US-021 |
| `rex/replay.py` | 11, 36, 78, 89, 118, 127 | Module docstring and inline comments declaring stub/placeholder result returns | **production-blocker** | US-020 |
| `rex/skills/trainer.py` | 127 | `# TODO: implement {name}` | **production-blocker** | US-022 |

### Dev-Only / Documented (no action required)

| File | Lines | Marker / Excerpt | Classification | Rationale |
|------|-------|-----------------|----------------|-----------|
| `archived/shopping_pwa/shopping_pwa.py` | 107, 176, 177, 178 | HTML `placeholder="..."` form attributes | **dev-only-documented** | Archived module (`archived/`); HTML input placeholder attributes are UI hints, not code stubs |
| `docs/archive/housekeeping/CODEX_REPO_AUDIT_ISSUES.json` | 302 | `"detail": "Replay reconstructs a ToolCall and returns a placeholder result…"` | **dev-only-documented** | Archived audit document; describes the replay stub for historical context |
| `scripts/claude_auditor.ps1` | 17 | `mark the task as TODO again` | **dev-only-documented** | Developer agent orchestration script; "TODO" here is a task-tracker state, not a code stub |
| `scripts/run_verifier_agent.ps1` | 42 | `mark the task back to TODO` | **dev-only-documented** | Same as above |
| `voice_loop.py` | 429 | `# Create listener with a placeholder loop` | **dev-only-documented** | Root-level legacy re-export shim; comment refers to initial event-loop value, not an incomplete implementation |

### False Positives

| File | Lines | Marker / Excerpt | Classification | Rationale |
|------|-------|-----------------|----------------|-----------|
| `rex/contracts/core.py` | 106, 110 | `with a redacted placeholder` / `redacted_value: The placeholder string` | **false-positive** | Docstring describing a redaction API parameter; "placeholder" is the intended value name |
| `rex/integrations/calendar/backends/ics_feed.py` | 231, 232 | `# File truncated inside a VEVENT — skip` / `truncated VEVENT block at end of file` | **false-positive** | Legitimate file-truncation detection for malformed ICS feeds |
| `rex/memory_utils.py` | 189 | `return norm.endswith(placeholder) or …` | **false-positive** | `placeholder` is a local function parameter name in a path-comparison utility |
| `rex/tools/registry.py` | 154 | `"""Placeholder handler for tools that delegate to another executor at runtime."""` | **false-positive** | `_noop_handler` is intentional runtime delegation design; it returns `{}` as a null stub for tools whose executor is registered separately at startup |
| `rex/tts_utils.py` | 46, 50, 58, 65, 106 | `_ABBREV_PLACEHOLDER = "\x00"` and usages | **false-positive** | Sentinel character (`\x00`) used to protect abbreviation periods during text splitting; named with `_ABBREV_PLACEHOLDER` by convention |
| `rex/voice/transcripts.py` | 72, 73, 196, 208, 216, 344 | `_ABBREV_PLACEHOLDER = "\x00"` and usages | **false-positive** | Same sentinel pattern as `rex/tts_utils.py` |
| `gui/package-lock.json` | 2656, 2658, 4089 | `"node_modules/cli-truncate"` / `cli-truncate` entries | **false-positive** | `cli-truncate` is an npm dependency; "truncate" in the package name matches the TRUNCAT pattern but is not a code stub |
| `tests/test_config_migration.py` | 182, 185 | `"""A truncated JSON file … is recovered from gracefully."""` / `# Simulate a truncated JSON write` | **false-positive** | Test for file-truncation recovery; "truncated" describes a real failure scenario under test |
| `tests/test_contracts_core.py` | 323 | `"""Should support custom redaction placeholder."""` | **false-positive** | Test docstring describing the redaction-value parameter |
| `tests/test_first_run.py` | 30, 51, 85, 129, 144 | `[test-first-run-jwt-placeholder]`, `[test-setup-password-placeholder]`, `[invalid-setup-token-placeholder]`, `[wrong-password-placeholder]` | **false-positive** | Intentionally-named test fixture strings using bracket notation to prevent accidental use as real secrets |
| `tests/test_log002_log_viewer.py` | 14 | `[test-log002-jwt-placeholder-32-bytes-min]` | **false-positive** | Test fixture JWT secret |
| `tests/test_openclaw_rex_voice_loop_text_mode.py` | 40 | `MagicMock(),  # placeholder assistant — will be overwritten below` | **false-positive** | Inline test comment; MagicMock is replaced before use |
| `tests/test_openclaw_voice_loop_optimized_text_mode.py` | 40 | `MagicMock(),  # placeholder — overwritten below` | **false-positive** | Same pattern |
| `tests/test_rr007_setup_register_protection.py` | 28, 48, 68, 115, 122, 123, 130, 142, 148 | Various `[test-rr007-…-placeholder]` strings | **false-positive** | Test fixture strings |
| `tests/test_rr008_log_auth.py` | 26, 60, 65 | `[test-rr008-jwt-placeholder-32-bytes-min]`, `[test-log-auth-password-placeholder]` | **false-positive** | Test fixture strings |
| `tests/test_rr010_ha_secret_required.py` | 16, 94, 95, 102, 104, 110, 116, 120, 126 | Various `[test-ha-…-placeholder]` strings | **false-positive** | Test fixture strings |
| `tests/test_us035_plugin_execution.py` | 270 | `assert "truncated" in result` | **false-positive** | Asserts that output-size limiting produces a "truncated" marker in the result |
| `tests/test_us042_ha_api_connection.py` | 15, 61, 82, 89, 116, 129, 135, 148, 233, 254 | Various `[test-ha-…-placeholder]` and `[test-bearer-token-placeholder]` strings | **false-positive** | Test fixture strings |
| `tests/test_us053_secret_management.py` | 130 | `# The example should only have placeholder values` | **false-positive** | Comment describing expected fixture contents |
| `tests/test_us099_unit_test_gaps_planner_registry_workflow.py` | 592 | `WorkflowStep(description="placeholder")` | **false-positive** | Minimal test data object; `"placeholder"` is a valid description value for testing |
| `tests/test_us315_real_data_pages.py` | 2 | `US-315: Replace placeholder data in Calendar, Email, and SMS pages` | **false-positive** | Module docstring describing the story under test; refers to a completed migration |

---

## Findings Not Flagged (Confirmed Clean)

- **Merge conflict markers:** None found.
- **Exposed secrets / API keys:** None found.
- **Documentation findings (132):** All informational; PRD and changelog references to "placeholder" are historical context, not production code stubs.

---

## Summary

| Category | Count | Status |
|----------|-------|--------|
| Production blockers (open) | 3 (entries in 3 files) | US-020, US-021, US-022 |
| Dev-only / documented | 5 entries | No action |
| False positives | 74 entries | No action |
| Secrets | 0 | Clean |
| Merge markers | 0 | Clean |
