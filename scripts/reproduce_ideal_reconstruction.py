from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msf_spi.ideal.patterns import load_angle_library
from msf_spi.ideal.simulation import run_ideal_reconstruction
from msf_spi.metrics import normalize_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the ideal pixel-replicated MSF-SPI reconstruction."
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "ideal_model.json")
    parser.add_argument(
        "--angle-library",
        type=Path,
        default=PROJECT_ROOT / "data" / "angle_library" / "optimal_angle_differences.json",
    )
    parser.add_argument("--target", type=Path, default=PROJECT_ROOT / "data" / "targets" / "USAF-1951.jpg")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "ideal_reconstruction")
    parser.add_argument("--max-points", type=int)
    return parser.parse_args()


def read_grayscale(path: Path, size: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"unable to read target image: {path}")
    return normalize_image(cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA))


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    slm_pixels = int(config["slm_pixels"])
    size = int(config.get("target_size", slm_pixels * int(config.get("oversample", 1))))
    maximum_radius = float(config.get("maximum_target_frequency", config.get("msf_radius", size / 2)))
    coefficients = tuple(config.get("eq10_coefficients", [2.2326, -1.1163, -1.1163, -0.5928, 1.3546]))
    image = read_grayscale(args.target, size)
    result = run_ideal_reconstruction(
        image,
        load_angle_library(args.angle_library),
        slm_pixels=slm_pixels,
        maximum_radius=maximum_radius,
        angle_increment_deg=float(config.get("angle_quantization_deg", 0.001)),
        coefficients=coefficients,
        regularization=float(config.get("sparse_regularization", 0.1)),
        max_points=args.max_points,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    panels = (
        (image, "(a) Target"),
        (result["direct_reconstruction"], "(b) Per-frequency response correction"),
        (result["sparse_reconstruction"], "(c) Sparse response-matrix correction"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (values, title) in zip(axes, panels):
        axis.imshow(values, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(args.output / "comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
    summary = {
        "model": "ideal pixel-replicated MSF-SPI",
        "frequency_points": result["frequency_points"],
        "phase_measurements": result["phase_measurements"],
        "angle_increment_deg": result["angle_increment_deg"],
        "direct_metrics": result["direct_metrics"],
        "sparse_metrics": result["sparse_metrics"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
