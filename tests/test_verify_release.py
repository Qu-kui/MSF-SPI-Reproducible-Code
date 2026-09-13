import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_release_verifier_passes_from_outside_repository(tmp_path):
    report = tmp_path / "release_report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_release.py"),
            "--quick",
            "--report",
            str(report),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    names = {row["name"] for row in payload["checks"]}
    assert {
        "package_import",
        "focused_tests",
        "quick_workflows",
        "source_integrity",
    } <= names
