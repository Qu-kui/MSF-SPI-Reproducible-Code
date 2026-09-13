from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "reference" / "angular_quantization" / "summary.json"


def clean(value):
    if isinstance(value, dict):
        return {
            key: clean(item)
            for key, item in value.items()
            if key not in {"diagnostic_path", "source_hashes_before_run", "source_hashes_after_run"}
        }
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, str):
        value = value.replace("ideal direct-mask Mode-4 MSF-SPI", "ideal algebraic MSF-SPI")
        if ":\\" in value:
            return Path(value).name
    return value


def main() -> None:
    payload = clean(json.loads(SOURCE.read_text(encoding="utf-8")))
    payload["calibration"]["angle_library_path"] = (
        "data/angle_library/optimal_angle_differences.json"
    )
    SOURCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
