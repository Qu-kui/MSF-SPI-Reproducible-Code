import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_quick_validation_runs_all_four_workflows(tmp_path):
    report = tmp_path / "validation_report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_quick_validation.py"),
            "--output",
            str(tmp_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert {row["name"] for row in payload["workflows"]} == {
        "bandwidth",
        "physical_comparison",
        "robustness",
        "ideal_reconstruction",
    }
