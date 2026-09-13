from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from msf_spi.physical.comparison import run_fmax_comparison
from msf_spi.physical.config import PhysicalSLMConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_image(path: Path, size: int):
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise FileNotFoundError(f"test image not found or unreadable: {path}") from error
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"test image not found or unreadable: {path}")
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA) / 255.0


def _write_outputs(result: dict, config: PhysicalSLMConfig, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    panels = (
        ("(a) Ground truth", result["reference"]),
        ("(b) Ideal circularly band-limited reference", result["ideal_msf_reference"]),
        ("(c) Conventional Fourier SPI", result["spi_reconstruction"]),
        ("(d) MSF-SPI with sparse response correction", result["msf_sparse_reconstruction"]),
    )
    metric_keys = (None, "msf_vs_ideal_metrics", "spi_metrics", "msf_sparse_metrics")
    figure, axes = plt.subplots(2, 2, figsize=(10, 10))
    for axis, (title, image), metric_key in zip(axes.ravel(), panels, metric_keys):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        if metric_key is not None:
            metrics = result[metric_key]
            title += f"\nSSIM={metrics['ssim']:.3f}, PSNR={metrics['psnr_db']:.3f} dB"
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    keys = (
        "spi_metrics",
        "msf_metrics",
        "msf_sparse_metrics",
        "spi_vs_ideal_metrics",
        "msf_vs_ideal_metrics",
        "msf_sparse_vs_ideal_metrics",
        "spi_frequency_count",
        "msf_frequency_count",
        "low_half_plane_count",
        "high_half_plane_count",
        "unique_exposure_count",
        "naive_exposure_count",
        "fmax",
        "carrier",
        "iris_mm",
        "fill_factor",
        "calibration_method",
        "sparse_response",
    )
    metrics = {key: result[key] for key in keys}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "configuration.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "model": "pixel-resolved physical SLM",
                "frequency_decomposition": "collinear",
                "angular_difference_deg": 0.0,
                "simulation_only": True,
                "artifacts": ["comparison.png", "metrics.json", "configuration.json"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the physical SPI and MSF-SPI comparison."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "physical_slm_256.json",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "targets" / "USAF-1951.jpg",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--response-matrix", type=Path, default=None)
    parser.add_argument("--rebuild-response", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    if args.verify_only:
        required = ("comparison.png", "metrics.json", "configuration.json", "run_manifest.json")
        missing = [name for name in required if not (output / name).is_file()]
        if missing:
            print("Missing comparison artifacts:", *missing, sep="\n  ")
            return 1
        print("Physical comparison artifacts are present.")
        return 0

    config = PhysicalSLMConfig.from_mapping(
        json.loads(args.config.resolve().read_text(encoding="utf-8"))
    )
    size = config.slm_pixels * config.oversample
    image = _load_image(args.image.resolve(), size)
    result = run_fmax_comparison(
        image,
        config,
        fmax=config.conventional_radius,
        carrier=config.carrier_cycles_per_fov,
        iris_mm=config.first_order_iris_radius_mm,
        cache_dir=output / "cache",
        response_matrix_path=args.response_matrix,
        rebuild_response=args.rebuild_response,
        progress_every=0 if args.quiet else 25,
    )
    _write_outputs(result, config, output)
    print(json.dumps(result["msf_sparse_metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
