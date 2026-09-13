from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_physical_comparison_cli_writes_artifacts(tmp_path):
    source = ROOT / "data" / "targets" / "USAF-1951.jpg"
    assert source.is_file()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "reproduce_physical_comparison.py"),
            "--config",
            str(ROOT / "configs" / "quick_validation.json"),
            "--image",
            str(source),
            "--output",
            str(tmp_path),
            "--quiet",
        ],
        check=True,
        cwd=tmp_path,
    )
    for name in ("comparison.png", "metrics.json", "configuration.json", "run_manifest.json"):
        assert (tmp_path / name).is_file()
