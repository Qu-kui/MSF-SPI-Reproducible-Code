from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np


ANGLE_HALF_WIDTH_DEG = (0.0, 0.0034, 0.0115, 0.03, 0.07, 0.14)
MASK_FREQUENCY_PERCENT = (-1.0, -0.5, 0.0, 0.5, 1.0)
SLM_PHASE_RMS_PI = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
DETECTOR_AC_SNR_DB = (math.inf, 40.0, 30.0, 20.0, 10.0, 0.0)
CALIBRATION_RMS_PERCENT = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
DEFOCUS_MM = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
RANDOM_SEEDS = (260603, 260604, 260605)


@dataclass(frozen=True)
class SweepCase:
    family: str
    level: float
    seed: int | None
    label: str


def angle_errors(
    records: Iterable[Mapping[str, float]], half_width_deg: float, seed: int
) -> np.ndarray:
    """Draw one positioning error per frequency command, shared by its phases."""
    rows = list(records)
    if half_width_deg < 0:
        raise ValueError("angle half-width must be non-negative")
    output = np.zeros(len(rows), dtype=np.float32)
    if half_width_deg == 0:
        return output
    rng = np.random.default_rng(seed)
    draws: dict[tuple[float, float, float], float] = {}
    for index, row in enumerate(rows):
        if float(row["fx1"]) == 0.0:
            continue
        key = (
            round(float(row["fx1"]), 6),
            round(float(row["corrected_fx2"]), 6),
            round(float(row["target_theta"]), 6),
        )
        if key not in draws:
            draws[key] = float(rng.uniform(-half_width_deg, half_width_deg))
        output[index] = draws[key]
    return output


def _zero_mean_unit_rms(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64)
    output -= output.mean()
    rms = float(np.sqrt(np.mean(output**2)))
    if rms <= 0:
        return np.zeros_like(output)
    return output / rms


def static_slm_phase_map(size: int, rms_rad: float, seed: int) -> np.ndarray:
    if size <= 0 or rms_rad < 0:
        raise ValueError("invalid SLM phase-map arguments")
    if rms_rad == 0:
        return np.zeros((size, size), dtype=np.float32)
    rng = np.random.default_rng(seed)
    phase = _zero_mean_unit_rms(rng.standard_normal((size, size))) * rms_rad
    return phase.astype(np.float32)


def calibration_drift(rms_percent: float, seed: int, channels: int = 5) -> np.ndarray:
    if channels < 2 or rms_percent < 0:
        raise ValueError("invalid calibration-drift arguments")
    if rms_percent == 0:
        return np.zeros(channels, dtype=np.float64)
    rng = np.random.default_rng(seed)
    return _zero_mean_unit_rms(rng.standard_normal(channels)) * (rms_percent / 100.0)


def detector_noise(
    buckets: np.ndarray, snr_db: float, seed: int
) -> tuple[np.ndarray, float]:
    values = np.asarray(buckets, dtype=np.float64)
    if math.isinf(float(snr_db)):
        return values.copy(), 0.0
    sigma = float(np.std(values) / (10.0 ** (float(snr_db) / 20.0)))
    rng = np.random.default_rng(seed)
    return values + rng.normal(0.0, sigma, values.shape), sigma

