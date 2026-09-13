import json
import os
import subprocess
import sys
from pathlib import Path


def test_ideal_cli_runs_from_an_unrelated_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "ideal"
    command = [
        sys.executable,
        str(root / "scripts" / "reproduce_ideal_reconstruction.py"),
        "--config",
        str(root / "configs" / "quick_validation.json"),
        "--angle-library",
        str(root / "data" / "angle_library" / "optimal_angle_differences.json"),
        "--output",
        str(output),
        "--max-points",
        "6",
    ]
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["frequency_points"] == 6
    assert (output / "comparison.png").is_file()
