from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import PhysicalSLMConfig as PhysicalBandwidthConfig
from .order_model import (
    normalize_phase_sequence,
    simulate_recovered_pattern,
    sinusoid_quality,
)


def coarse_frequency_schedule(cfg: PhysicalBandwidthConfig) -> tuple[float, ...]:
    frequencies = tuple(float(value) for value in cfg.coarse_target_values)
    if cfg.slm_pixels < 256:
        return frequencies
    return tuple(sorted(set((4.0, 8.0, 12.0) + frequencies)))


def compare_convergence_rows(
    base_rows: Iterable[dict], high_rows: Iterable[dict], *, tolerance: float = 0.01
) -> dict:
    """Compare matched boundary cases across numerical sampling levels."""
    keys = ("target_frequency", "angle_deg", "phase_deg")

    def indexed(rows):
        return {
            tuple(round(float(row[key]), 9) for key in keys): row for row in rows
        }

    base = indexed(base_rows)
    high = indexed(high_rows)
    if not base or set(base) != set(high):
        return {
            "status": "sampling_sensitive",
            "reason": "unmatched convergence cases",
            "case_count": len(set(base) & set(high)),
        }
    metrics = ("correlation", "leakage", "modulation_error")
    maxima = {
        metric: max(abs(float(base[key][metric]) - float(high[key][metric])) for key in base)
        for metric in metrics
    }
    classification_changes = sum(
        bool(base[key]["operational_passes"]) != bool(high[key]["operational_passes"])
        for key in base
    )
    converged = classification_changes == 0 and all(
        value <= tolerance for value in maxima.values()
    )
    return {
        "status": "converged" if converged else "sampling_sensitive",
        "case_count": len(base),
        "classification_changes": classification_changes,
        "metric_tolerance": float(tolerance),
        "max_absolute_metric_changes": maxima,
    }


def select_convergence_cases(
    rows: Iterable[dict], passing_frequency: float, failing_frequency: float
) -> tuple[tuple[float, float], ...]:
    """Select the worst passing angle and the strongest first-failure angle."""
    records = list(rows)
    passing = [
        row
        for row in records
        if np.isclose(float(row["target_frequency"]), passing_frequency)
        and bool(row["operational_passes"])
    ]
    failing = [
        row
        for row in records
        if np.isclose(float(row["target_frequency"]), failing_frequency)
        and not bool(row["operational_passes"])
    ]
    output: list[tuple[float, float]] = []
    if passing:
        worst = max(passing, key=lambda row: float(row["leakage"]))
        output.append((float(passing_frequency), float(worst["angle_deg"])))
    if failing:
        worst = max(failing, key=lambda row: float(row["leakage"]))
        case = (float(failing_frequency), float(worst["angle_deg"]))
        if case not in output:
            output.append(case)
    return tuple(output)


def build_run_configuration(cfg: PhysicalBandwidthConfig) -> dict:
    """Return the reviewer-facing assumptions for a bandwidth-only run."""
    values = asdict(cfg)
    return {
        **values,
        "active_width_m": cfg.active_width_m,
        "order_lens_na": cfg.order_lens_na,
        "relay_na": cfg.relay_na,
        "separate_slm_and_difference_measurements": True,
        "full_spi_msf_reconstruction_enabled": False,
        "simulation_only": True,
    }


def summarize_frequency_rows(
    rows: Iterable[dict],
    *,
    expected_frequencies: Sequence[float],
    expected_angles: Sequence[float],
    expected_phases: Sequence[float],
) -> dict:
    """Summarize continuous all-case bandwidth for two fixed criteria."""
    records = list(rows)
    required = {
        (round(float(angle), 9), round(float(phase), 9))
        for angle in expected_angles
        for phase in expected_phases
    }

    def bandwidth(pass_key: str) -> float:
        result = 0.0
        for frequency in expected_frequencies:
            selected = [
                row
                for row in records
                if np.isclose(float(row["target_frequency"]), float(frequency))
            ]
            observed = {
                (round(float(row["angle_deg"]), 9), round(float(row["phase_deg"]), 9))
                for row in selected
            }
            if observed != required or not all(bool(row[pass_key]) for row in selected):
                break
            result = float(frequency)
        return result

    operational = bandwidth("operational_passes")
    conservative = bandwidth("strict_passes")
    near_threshold = None
    for frequency in expected_frequencies:
        if float(frequency) <= operational:
            continue
        selected = [
            row
            for row in records
            if np.isclose(float(row["target_frequency"]), float(frequency))
        ]
        if selected:
            near_threshold = {
                "frequency": float(frequency),
                "failed_cases": sum(not bool(row["operational_passes"]) for row in selected),
                "case_count": len(selected),
            }
        break
    return {
        "conservative_bandwidth": conservative,
        "operational_bandwidth": operational,
        "near_threshold": near_threshold,
        "row_count": len(records),
    }


def evaluate_frequency_dual(
    cfg: PhysicalBandwidthConfig,
    *,
    target_frequency: float,
    carrier_frequency: float,
    iris_radius_mm: float,
    angles_deg: Sequence[float],
    phases_deg: Sequence[float],
    stage: str,
) -> list[dict]:
    """Propagate one quartet per angle and score two declared quality tiers."""
    quartet_phases = (0.0, 90.0, 180.0, 270.0)
    requested = tuple(float(value) for value in phases_deg)
    if not set(requested).issubset(quartet_phases):
        raise ValueError("phases must be selected from the four-step quartet")
    rows: list[dict] = []
    for angle in angles_deg:
        simulations = [
            simulate_recovered_pattern(
                float(target_frequency),
                float(angle),
                phase,
                float(carrier_frequency),
                float(iris_radius_mm),
                cfg,
            )
            for phase in quartet_phases
        ]
        normalized = normalize_phase_sequence(
            np.stack([item.recovered_intensity for item in simulations])
        )
        for phase in requested:
            index = quartet_phases.index(phase)
            simulation = simulations[index]
            strict = sinusoid_quality(
                simulation.target_intensity,
                normalized[index],
                float(target_frequency),
                float(angle),
                phase,
                cfg.thresholds,
            )
            operational = sinusoid_quality(
                simulation.target_intensity,
                normalized[index],
                float(target_frequency),
                float(angle),
                phase,
                cfg.operational_thresholds,
            )
            rows.append(
                {
                    "stage": str(stage),
                    "target_frequency": float(target_frequency),
                    "angle_deg": float(angle),
                    "phase_deg": phase,
                    "carrier": float(carrier_frequency),
                    "iris_mm": float(iris_radius_mm),
                    "fill_factor": float(cfg.fill_factor),
                    "correlation": strict.correlation,
                    "leakage": strict.leakage,
                    "modulation_error": strict.modulation_error,
                    "phase_error_deg": strict.fitted_phase_error_deg,
                    "strict_passes": strict.passes,
                    "operational_passes": operational.passes,
                    "selected_efficiency": simulation.diagnostics.selected_efficiency,
                }
            )
    return rows


def _config_hash(cfg: PhysicalBandwidthConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_bandwidth_stage(
    cfg: PhysicalBandwidthConfig,
    output_dir: Path,
    *,
    resume: bool = True,
    refine: bool = False,
) -> dict:
    """Run a bandwidth-only scale-matched search for one fill factor."""
    cfg.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "bandwidth_rows.jsonl"
    summary_path = output / "bandwidth_summary.json"
    config_hash = _config_hash(cfg)
    rows: list[dict] = []
    completed: set[tuple[float, float, float, str]] = set()
    if resume and rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("config_hash") != config_hash:
                raise ValueError("cached 256-bandwidth rows use a different configuration")
            rows.append(row)
            completed.add(
                (
                    float(row["carrier"]),
                    float(row["iris_mm"]),
                    float(row["target_frequency"]),
                    str(row["stage"]),
                )
            )

    carrier = float(cfg.coarse_carrier_values[0])
    iris_mm = float(cfg.coarse_iris_values_mm[0])
    mode = "a" if resume else "w"
    with rows_path.open(mode, encoding="utf-8") as handle:
        for frequency in coarse_frequency_schedule(cfg):
            stage = (
                "low_frequency_anchor"
                if float(frequency) < float(cfg.coarse_target_values[0])
                else "scale_matched"
            )
            key = (carrier, iris_mm, float(frequency), stage)
            if key in completed:
                continue
            new_rows = evaluate_frequency_dual(
                cfg,
                target_frequency=float(frequency),
                carrier_frequency=carrier,
                iris_radius_mm=iris_mm,
                angles_deg=cfg.coarse_angle_values_deg,
                phases_deg=cfg.phases_deg,
                stage=stage,
            )
            for row in new_rows:
                row["config_hash"] = config_hash
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
            rows.extend(new_rows)

    scale_rows = [
        row
        for row in rows
        if row["stage"] in {"low_frequency_anchor", "scale_matched"}
    ]
    coarse_aggregate = summarize_frequency_rows(
        scale_rows,
        expected_frequencies=coarse_frequency_schedule(cfg),
        expected_angles=cfg.coarse_angle_values_deg,
        expected_phases=cfg.phases_deg,
    )
    aggregate = dict(coarse_aggregate)
    if refine:
        coarse_boundary = float(coarse_aggregate["operational_bandwidth"])
        if coarse_boundary <= 0:
            coarse_boundary = float(cfg.coarse_target_values[0])
        step = float(cfg.boundary_step)
        start = max(float(cfg.coarse_target_values[0]), coarse_boundary - 1.0)
        stop = coarse_boundary + 4.0
        refinement_frequencies = tuple(
            float(value) for value in np.arange(start, stop + 0.5 * step, step)
        )
        with rows_path.open("a", encoding="utf-8") as handle:
            for frequency in refinement_frequencies:
                key = (carrier, iris_mm, frequency, "boundary_refinement")
                existing = [
                    row
                    for row in rows
                    if row["stage"] == "boundary_refinement"
                    and np.isclose(float(row["target_frequency"]), frequency)
                ]
                if key in completed:
                    new_rows = existing
                else:
                    print(
                        f"  boundary refinement: fill={cfg.fill_factor:.2f}, "
                        f"f={frequency:g} cycles/FOV",
                        flush=True,
                    )
                    new_rows = evaluate_frequency_dual(
                        cfg,
                        target_frequency=frequency,
                        carrier_frequency=carrier,
                        iris_radius_mm=iris_mm,
                        angles_deg=cfg.final_angle_values_deg,
                        phases_deg=cfg.phases_deg,
                        stage="boundary_refinement",
                    )
                    for row in new_rows:
                        row["config_hash"] = config_hash
                        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                    handle.flush()
                    rows.extend(new_rows)
                    completed.add(key)
                if new_rows and not all(bool(row["operational_passes"]) for row in new_rows):
                    break
        refinement_rows = [row for row in rows if row["stage"] == "boundary_refinement"]
        evaluated_frequencies = tuple(
            sorted({float(row["target_frequency"]) for row in refinement_rows})
        )
        if evaluated_frequencies:
            aggregate = summarize_frequency_rows(
                refinement_rows,
                expected_frequencies=evaluated_frequencies,
                expected_angles=cfg.final_angle_values_deg,
                expected_phases=cfg.phases_deg,
            )
            aggregate["conservative_bandwidth"] = coarse_aggregate[
                "conservative_bandwidth"
            ]
            aggregate["coarse_summary"] = coarse_aggregate
            aggregate["refinement_frequencies"] = list(evaluated_frequencies)
    status = (
        "bandwidth_established"
        if aggregate["operational_bandwidth"] >= float(cfg.coarse_target_values[0])
        else "no_usable_operational_band"
    )
    summary = {
        "status": status,
        "configuration": build_run_configuration(cfg),
        "fill_factor_results": {str(cfg.fill_factor): aggregate},
        "selected_pair": {"carrier": carrier, "iris_mm": iris_mm},
        **aggregate,
        "convergence": {
            "status": "not_requested" if not refine else "base_sampling_complete",
            "refinement_requested": bool(refine),
            "higher_sampling_check": "not_run",
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
