from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_robustness_cli_writes_verified_quick_artifacts(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "reproduce_robustness.py"),
            "--quick",
            "--output",
            str(tmp_path),
        ],
        check=True,
        cwd=tmp_path,
    )
    for name in (
        "robustness_summary.png",
        "summary_metrics.json",
        "error_realizations.json",
        "verification_report.json",
    ):
        assert (tmp_path / name).is_file()
