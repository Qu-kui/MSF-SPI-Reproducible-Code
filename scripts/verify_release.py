from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def run_subprocess(name: str, command: list[str], *, cwd: Path) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return {
        "name": name,
        "status": "passed" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "duration_seconds": time.perf_counter() - started,
        "command": command,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def verify_source_integrity() -> dict:
    started = time.perf_counter()
    manifest_path = ROOT / "docs" / "source_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    failures = []
    for entry in payload["entries"]:
        source = Path(entry["source"])
        if not source.is_file():
            failures.append({"source": str(source), "reason": "missing"})
            continue
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            failures.append(
                {
                    "source": str(source),
                    "reason": "sha256 mismatch",
                    "expected": entry["sha256"],
                    "actual": digest,
                }
            )
    return {
        "name": "source_integrity",
        "status": "passed" if not failures else "failed",
        "duration_seconds": time.perf_counter() - started,
        "source_count": len(payload["entries"]),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the reproducible MSF-SPI release.")
    parser.add_argument("--quick", action="store_true", help="run reduced CPU workflows")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks = []
    checks.append(
        run_subprocess(
            "package_import",
            [sys.executable, "-c", "import msf_spi; print(msf_spi.__version__)"],
            cwd=report_path.parent,
        )
    )
    focused = [
        "tests/test_configuration.py",
        "tests/test_reconstruction_primitives.py",
        "tests/test_reference_artifacts.py",
        "tests/test_documentation.py",
    ]
    checks.append(
        run_subprocess(
            "focused_tests",
            [sys.executable, "-m", "pytest", "-q", *focused],
            cwd=ROOT,
        )
    )
    if args.quick:
        checks.append(
            run_subprocess(
                "quick_workflows",
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_quick_validation.py"),
                    "--output",
                    str(report_path.parent / "quick_workflows"),
                ],
                cwd=report_path.parent,
            )
        )
    checks.append(verify_source_integrity())
    status = "passed" if all(row["status"] == "passed" for row in checks) else "failed"
    payload = {"status": status, "repository": str(ROOT), "checks": checks}
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
