from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_bandwidth_cli_writes_summary(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_slm_bandwidth.py"),
            "--config",
            str(ROOT / "configs" / "quick_validation.json"),
            "--output",
            str(tmp_path),
        ],
        check=True,
        cwd=tmp_path,
    )
    assert (tmp_path / "bandwidth_rows.jsonl").is_file()
    assert (tmp_path / "bandwidth_summary.json").is_file()
