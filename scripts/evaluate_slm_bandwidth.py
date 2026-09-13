from __future__ import annotations

import argparse
import json
from pathlib import Path

from msf_spi.physical.bandwidth import run_bandwidth_stage
from msf_spi.physical.config import PhysicalSLMConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> PhysicalSLMConfig:
    return PhysicalSLMConfig.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the usable spatial-frequency range of the physical SLM model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "physical_slm_256.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    rows_path = output / "bandwidth_rows.jsonl"
    summary_path = output / "bandwidth_summary.json"
    if args.verify_only:
        missing = [str(path) for path in (rows_path, summary_path) if not path.is_file()]
        if missing:
            print("Missing bandwidth artifacts:", *missing, sep="\n  ")
            return 1
        print("Bandwidth artifacts are present.")
        return 0

    config = load_config(args.config.resolve())
    summary = run_bandwidth_stage(
        config,
        output,
        resume=args.resume,
        refine=args.refine,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

