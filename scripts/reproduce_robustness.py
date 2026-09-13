from __future__ import annotations

import argparse
import json
from pathlib import Path

from msf_spi.robustness.artifacts import (
    verify_robustness_artifacts,
    write_robustness_artifacts,
)
from msf_spi.robustness.pipeline import run_robustness_pipeline


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the six robustness sweeps using the final sparse-response-corrected "
            "physical MSF-SPI reconstruction."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--image",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "targets" / "USAF-1951.jpg",
    )
    parser.add_argument("--response-matrix", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    if args.verify_only:
        report = verify_robustness_artifacts(output)
        print(json.dumps(report, indent=2))
        return 0 if report["complete"] else 1

    result = run_robustness_pipeline(
        output,
        quick=args.quick,
        resume=args.resume,
        image_path=args.image.resolve(),
        response_matrix_source=args.response_matrix,
    )
    manifest = write_robustness_artifacts(result, output)
    report = verify_robustness_artifacts(output)
    print(json.dumps({"manifest": manifest, "verification": report}, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

