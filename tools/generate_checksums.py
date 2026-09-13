from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "precomputed" / "checksums.sha256"


def main() -> None:
    selected = [ROOT / "data" / "targets" / "USAF-1951.jpg"]
    selected.extend((ROOT / "data" / "angle_library").glob("*"))
    selected.extend((ROOT / "data" / "precomputed").glob("*"))
    selected.extend((ROOT / "results" / "reference").rglob("*"))
    selected = sorted(
        path for path in selected if path.is_file() and path.resolve() != OUTPUT.resolve()
    )
    lines = []
    for path in selected:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Wrote {len(lines)} checksums to {OUTPUT}")


if __name__ == "__main__":
    main()
