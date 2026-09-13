from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def run(name: str, command: list[str], output: Path) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=output, capture_output=True, text=True)
    return {
        "name": name,
        "status": "passed" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "duration_seconds": time.perf_counter() - started,
        "command": command,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all reduced CPU validation workflows.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    commands = [
        (
            "bandwidth",
            [python, str(ROOT / "scripts" / "evaluate_slm_bandwidth.py"), "--config", str(ROOT / "configs" / "quick_validation.json"), "--output", str(output / "bandwidth")],
        ),
        (
            "physical_comparison",
            [python, str(ROOT / "scripts" / "reproduce_physical_comparison.py"), "--config", str(ROOT / "configs" / "quick_validation.json"), "--output", str(output / "physical_comparison"), "--quiet"],
        ),
        (
            "robustness",
            [python, str(ROOT / "scripts" / "reproduce_robustness.py"), "--quick", "--output", str(output / "robustness")],
        ),
        (
            "ideal_reconstruction",
            [python, str(ROOT / "scripts" / "reproduce_ideal_reconstruction.py"), "--config", str(ROOT / "configs" / "quick_validation.json"), "--angle-library", str(ROOT / "data" / "angle_library" / "optimal_angle_differences.json"), "--output", str(output / "ideal_reconstruction"), "--max-points", "6"],
        ),
    ]
    workflows = []
    for name, command in commands:
        workflow = run(name, command, output)
        workflows.append(workflow)
        if workflow["status"] != "passed":
            break
    report = {
        "status": "passed" if len(workflows) == len(commands) and all(row["status"] == "passed" for row in workflows) else "failed",
        "workflows": workflows,
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
