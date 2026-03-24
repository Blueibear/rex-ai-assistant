"""Generate coverage gap analysis from test-audit-coverage.txt."""

import json
import re
from pathlib import Path


def main():
    # Parse coverage report
    cov_data = {}
    with open("test-audit-coverage.txt") as f:
        for line in f:
            m = re.match(r"^(rex[/\\].+?\.py)\s+\d+\s+\d+\s+(\d+)%", line)
            if m:
                path = m.group(1).replace("\\", "/")
                pct = int(m.group(2))
                cov_data[path] = pct

    # Get all test files
    test_files = {f.stem.replace("test_", "") for f in Path("tests").glob("test_*.py")}

    # Priority mapping for key modules
    high_priority = {
        "rex/assistant.py",
        "rex/cli.py",
        "rex/config.py",
        "rex/llm_client.py",
        "rex/openclaw/tool_executor.py",
        "rex/app.py",
        "rex/__main__.py",
        "rex/voice_loop.py",
        "rex/integrations.py",
        "rex/wake_acknowledgment.py",
        "rex/compat.py",
        "rex/capabilities/__init__.py",
    }

    medium_priority = {
        "rex/vscode_service.py",
        "rex/plugin_loader.py",
        "rex/ha_bridge.py",
        "rex/followup_engine.py",
        "rex/reminder_service.py",
        "rex/wakeword/embedding.py",
        "rex/browser_automation.py",
        "rex/service_supervisor.py",
        "rex/integrations/email_service.py",
        "rex/integrations/calendar_service.py",
        "rex/integrations/sms_service.py",
        "rex/windows_service.py",
        "rex/executor.py",
        "rex/startup.py",
    }

    gaps = []
    for path, pct in sorted(cov_data.items(), key=lambda x: x[1]):
        module_name = (
            path.replace("rex/", "").replace("/", "_").replace(".py", "").replace("\\", "_")
        )
        has_test = any(t in module_name or module_name in t for t in test_files)

        if path in high_priority:
            priority = "high"
        elif path in medium_priority or pct < 50:
            priority = "medium"
        elif pct < 75:
            priority = "medium"
        else:
            priority = "low"

        gaps.append(
            {
                "module_path": path,
                "current_coverage_pct": pct,
                "has_test_file": has_test,
                "priority": priority,
            }
        )

    # Filter to only interesting ones (< 75% or no test file)
    interesting = [g for g in gaps if g["current_coverage_pct"] < 75 or not g["has_test_file"]]

    with open("test-audit-coverage-gaps.json", "w") as f:
        json.dump(interesting, f, indent=2)

    print(f"Total modules tracked: {len(gaps)}")
    print(f"Modules with gaps (< 75% or no test): {len(interesting)}")
    print()
    print("0% coverage modules:")
    for g in gaps:
        if g["current_coverage_pct"] == 0:
            print(f"  {g['module_path']} - priority: {g['priority']}")
    print()
    print("High priority gaps:")
    for g in interesting:
        if g["priority"] == "high":
            print(f"  {g['module_path']} - {g['current_coverage_pct']}%")
    print()
    print("Medium priority gaps (top 10):")
    med = [g for g in interesting if g["priority"] == "medium"]
    for g in med[:10]:
        print(f"  {g['module_path']} - {g['current_coverage_pct']}%")


if __name__ == "__main__":
    main()
