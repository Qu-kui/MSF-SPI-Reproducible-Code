from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..metrics import image_metrics
from .cases import RobustnessCase, build_robustness_cases


FAMILY_TITLES = {
    "mask_angle": "Mask angle",
    "mask_frequency": "Mask frequency",
    "slm_static_phase": "SLM static phase",
    "detector_noise": "Detector noise",
    "calibration_mismatch": "Calibration mismatch",
    "conjugate_defocus": "Conjugate defocus",
}


def _safe(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return str(value).replace("-", "m").replace("+", "p").replace(".", "p")


def _case_name(case: RobustnessCase) -> str:
    return f"level_{_safe(case.level)}_seed_{_safe(case.seed)}.png"


def _save_gray(path: Path, image: np.ndarray) -> None:
    values = np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", values)
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    encoded.tofile(path)


def _json_metric(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isinf(value):
        return "Infinity"
    return value


def _metric_rows(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        case = row["case"]
        output.append(
            {
                "family": case.family,
                "level": case.level,
                "unit": case.unit,
                "seed": case.seed,
                "label": case.label,
                "ssim_full": row["metrics_full"]["ssim"],
                "psnr_full_db": row["metrics_full"]["psnr_db"],
                "ssim_matched_r64": row["metrics_matched"]["ssim"],
                "psnr_matched_r64_db": row["metrics_matched"]["psnr_db"],
            }
        )
    return output


def _group_rows(rows: list[dict]) -> list[dict]:
    levels = []
    for row in rows:
        level = float(row["level"])
        if level not in levels:
            levels.append(level)
    groups = []
    for level in levels:
        subset = [row for row in rows if float(row["level"]) == level]
        groups.append(
            {
                "level": level,
                "label": subset[0]["label"],
                **{
                    f"mean_{key}": float(np.mean([float(item[key]) for item in subset]))
                    for key in (
                        "ssim_full",
                        "psnr_full_db",
                        "ssim_matched_r64",
                        "psnr_matched_r64_db",
                    )
                },
                **{
                    f"std_{key}": float(np.std([float(item[key]) for item in subset]))
                    for key in (
                        "ssim_full",
                        "psnr_full_db",
                        "ssim_matched_r64",
                        "psnr_matched_r64_db",
                    )
                },
                "sample_count": len(subset),
            }
        )
    return groups


def _representatives(rows: list[dict]) -> list[dict]:
    output = []
    seen = set()
    baseline = next(row for row in rows if row["case"].is_baseline)
    output.append(baseline)
    seen.add(float(baseline["case"].level))
    for row in rows:
        level = float(row["case"].level)
        if level not in seen:
            output.append(row)
            seen.add(level)
    return output


def _comparison_figure(
    reference: np.ndarray,
    ideal: np.ndarray,
    rows: list[dict],
    path: Path,
) -> None:
    representatives = _representatives(rows)
    panels = [("Original", reference), ("Ideal R=64", ideal)] + [
        (row["case"].label, row["reconstruction"]) for row in representatives
    ]
    columns = 4
    lines = int(math.ceil(len(panels) / columns))
    fig, axes = plt.subplots(lines, columns, figsize=(3.5 * columns, 3.7 * lines))
    axes = np.asarray(axes).reshape(-1)
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[len(panels) :]:
        ax.axis("off")
    fig.suptitle(FAMILY_TITLES[rows[0]["case"].family], fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _curve_figure(groups: list[dict], family: str, path: Path) -> None:
    x = np.arange(len(groups))
    labels = [row["label"] for row in groups]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    definitions = (
        ("ssim_full", "SSIM vs complete reference"),
        ("psnr_full_db", "PSNR vs complete reference"),
        ("ssim_matched_r64", "SSIM vs matched ideal R=64"),
        ("psnr_matched_r64_db", "PSNR vs matched ideal R=64"),
    )
    for ax, (metric, title) in zip(axes.flat, definitions):
        mean = np.asarray([row[f"mean_{metric}"] for row in groups])
        std = np.asarray([row[f"std_{metric}"] for row in groups])
        ax.plot(x, mean, "o-", color="#1f77b4")
        if np.any(std > 0):
            ax.fill_between(x, mean - std, mean + std, color="#1f77b4", alpha=0.2)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.suptitle(FAMILY_TITLES[family])
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _summary_figure(summary: dict[str, list[dict]], path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
    for ax, (family, groups) in zip(axes.flat, summary.items()):
        x = np.arange(len(groups))
        mean = np.asarray([row["mean_ssim_matched_r64"] for row in groups])
        std = np.asarray([row["std_ssim_matched_r64"] for row in groups])
        ax.plot(x, mean, "o-", color="#0b6fa4")
        if np.any(std > 0):
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color="#0b6fa4")
        ax.set_xticks(x, [row["label"] for row in groups], rotation=25, ha="right")
        ax.set_title(FAMILY_TITLES[family])
        ax.set_ylabel("SSIM vs matched ideal R=64")
        ax.grid(alpha=0.25)
    fig.suptitle("256x256 physical-SLM MSF-SPI robustness", fontsize=17)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _baseline_comparison_figure(result: dict, path: Path) -> None:
    panels = (
        ("(a) Original", result["reference"], {"ssim": 1.0, "psnr_db": math.inf}),
        (
            "(b) Ideal band limit, R=32",
            result["ideal_spi_reference"],
            image_metrics(result["reference"], result["ideal_spi_reference"]),
        ),
        (
            "(c) Physical SPI, R=32",
            result["spi_baseline"]["reconstruction"],
            result["spi_baseline"]["metrics_full"],
        ),
        (
            "(d) Ideal band limit, R=64",
            result["ideal_msf_reference"],
            image_metrics(result["reference"], result["ideal_msf_reference"]),
        ),
        (
            "(e) Physical MSF-SPI, R=64",
            result["baseline"]["reconstruction"],
            result["baseline"]["metrics_full"],
        ),
    )
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.9), constrained_layout=True)
    for ax, (title, values, metrics) in zip(axes, panels):
        psnr = "inf" if math.isinf(metrics["psnr_db"]) else f"{metrics['psnr_db']:.3f} dB"
        ax.imshow(values, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"{title}\nSSIM={metrics['ssim']:.3f}, PSNR={psnr}", fontsize=9)
        ax.axis("off")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _supplementary_text(result: dict, summary: dict[str, list[dict]]) -> str:
    base_full = result["baseline"]["metrics_full"]
    base_matched = result["baseline"]["metrics_matched"]
    return f"""# Conservative supplementary draft

## English

Robustness was evaluated with a 256x256 phase-only SLM model using an 8-um
pixel pitch, 95% fill factor, a measured-like Gaussian incident amplitude,
finite Fourier-plane order selection, and a circular conventional bandwidth of
32 cycles/FOV. The MSF-SPI reconstruction radius was 64 cycles/FOV. A single
object-independent five-channel calibration and a sparse nominal system-response
matrix constructed from the illumination kernels were fixed before testing;
neither was repeated after an error was introduced. The response matrix jointly
accounts for residual inter-frequency coupling that is not removed by a scalar
per-pattern gain correction. Mask-angle positioning, mask-frequency fabrication, static SLM
phase, detector noise, channel-calibration mismatch, and sample-mask conjugate
defocus were varied independently. Stochastic cases use three fixed random
realizations, while defocus is calculated by angular-spectrum propagation.
The error-free reconstruction gives SSIM={base_full['ssim']:.4f} and
PSNR={base_full['psnr_db']:.3f} dB against the complete reference, and
SSIM={base_matched['ssim']:.4f} and PSNR={base_matched['psnr_db']:.3f} dB
against the matched radius-64 reference. These results are numerical
sensitivity evidence rather than experimental validation.

"""


def write_robustness_artifacts(result: dict, output_dir: Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _save_gray(output / "reference.png", result["reference"])
    _save_gray(output / "ideal_spi_r32.png", result["ideal_spi_reference"])
    _save_gray(output / "baseline_spi.png", result["spi_baseline"]["reconstruction"])
    _save_gray(output / "ideal_msf_r64.png", result["ideal_msf_reference"])
    _save_gray(output / "baseline_msf.png", result["baseline"]["reconstruction"])
    _baseline_comparison_figure(result, output / "baseline_comparison.png")
    summary = {}
    for family, rows in result["family_results"].items():
        folder = output / family
        recon_dir = folder / "reconstructions"
        recon_dir.mkdir(parents=True, exist_ok=True)
        metrics = _metric_rows(rows)
        for row in rows:
            _save_gray(recon_dir / _case_name(row["case"]), row["reconstruction"])
        with (folder / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
        (folder / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_metric),
            encoding="utf-8",
        )
        groups = _group_rows(metrics)
        summary[family] = groups
        (folder / "summary.json").write_text(
            json.dumps(groups, indent=2, ensure_ascii=False, default=_json_metric),
            encoding="utf-8",
        )
        _comparison_figure(
            result["reference"], result["ideal_msf_reference"], rows, folder / "comparison.png"
        )
        _curve_figure(groups, family, folder / "ssim_psnr_curves.png")
        manifest = {
            "family": family,
            "case_count": len(rows),
            "calibration_frozen": True,
            "separate_slm_and_difference_measurements": True,
        }
        (folder / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    _summary_figure(summary, output / "robustness_summary.png")
    (output / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_metric),
        encoding="utf-8",
    )
    (output / "configuration.json").write_text(
        json.dumps(result["configuration"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "calibration.json").write_text(
        json.dumps(
            {
                "direct_weights": result["global_weights"].tolist(),
                "diagnostics": result["weight_diagnostics"],
                "independent_validation": result["calibration_validation"],
                "system_response_matrix": result["response_matrix"],
                "frozen_for_all_error_cases": True,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output / "model_crosscheck.json").write_text(
        json.dumps(result["model_crosscheck"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "error_realizations.json").write_text(
        json.dumps(result["realizations"], indent=2, ensure_ascii=False, default=_json_metric),
        encoding="utf-8",
    )
    (output / "supplementary_draft.md").write_text(
        _supplementary_text(result, summary), encoding="utf-8"
    )
    manifest = {
        "status": "complete",
        "simulation_only": True,
        "families": {family: len(rows) for family, rows in result["family_results"].items()},
        "frequency_count": 1 + 2 * len(result["points"]),
        "spi_baseline_metrics_full": result["spi_baseline"]["metrics_full"],
        "msf_baseline_metrics_full": result["baseline"]["metrics_full"],
        "separate_slm_and_difference_measurements": True,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def verify_robustness_artifacts(output_dir: Path) -> dict:
    output = Path(output_dir)
    missing = []
    for name in (
        "reference.png",
        "ideal_spi_r32.png",
        "baseline_spi.png",
        "ideal_msf_r64.png",
        "baseline_msf.png",
        "baseline_comparison.png",
        "robustness_summary.png",
        "summary_metrics.json",
        "configuration.json",
        "calibration.json",
        "model_crosscheck.json",
        "error_realizations.json",
        "supplementary_draft.md",
        "run_manifest.json",
        "pipeline_cache.npz",
    ):
        path = output / name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
    for family, cases in build_robustness_cases().items():
        folder = output / family
        for name in (
            "comparison.png",
            "ssim_psnr_curves.png",
            "metrics.csv",
            "metrics.json",
            "summary.json",
            "manifest.json",
        ):
            path = folder / name
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
        images = list((folder / "reconstructions").glob("*.png")) if (folder / "reconstructions").exists() else []
        if len(images) != len(cases):
            missing.append(f"{family}: expected {len(cases)} reconstructions, found {len(images)}")
    report = {"complete": not missing, "missing": missing}
    (output / "verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
