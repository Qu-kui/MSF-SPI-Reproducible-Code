from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import re

import numpy as np


DEFAULT_EQ10_COEFFICIENTS = (2.2326, -1.1163, -1.1163, -0.5928, 1.3546)

_KEY_PATTERN = re.compile(
    r"fx1_(?P<slm>-?[\d.]+)_fx2_(?P<mask>-?[\d.]+)_"
    r"tfx_(?P<tfx>-?\d+)_tfy_(?P<tfy>-?\d+)_"
    r"angle_(?P<angle>-?[\d.]+)_p(?P<slm_phase>-?[\d.]+)_"
    r"(?P<mask_phase>-?[\d.]+)"
)


@dataclass(frozen=True)
class AngleLibraryEntry:
    """One phase-resolved frequency decomposition from the angle library."""

    slm_frequency: float
    mask_frequency: float
    target_fx: int
    target_fy: int
    target_angle_deg: float
    slm_phase_deg: float
    mask_phase_deg: float
    angular_difference_deg: float

    @property
    def total_phase_deg(self) -> float:
        return (self.slm_phase_deg + self.mask_phase_deg) % 360.0

    @property
    def point(self) -> tuple[int, int]:
        return self.target_fx, self.target_fy


def load_angle_library(path: str | Path) -> list[AngleLibraryEntry]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    values = payload.get("optimal_angle_diffs", payload)
    if not isinstance(values, dict) or not values:
        raise ValueError("the angle library must contain a non-empty mapping")
    entries: list[AngleLibraryEntry] = []
    for key, angular_difference in values.items():
        match = _KEY_PATTERN.fullmatch(str(key))
        if match is None:
            raise ValueError(f"unrecognized angle-library key: {key}")
        row = match.groupdict()
        entries.append(
            AngleLibraryEntry(
                slm_frequency=float(row["slm"]),
                mask_frequency=float(row["mask"]),
                target_fx=int(row["tfx"]),
                target_fy=int(row["tfy"]),
                target_angle_deg=float(row["angle"]),
                slm_phase_deg=float(row["slm_phase"]),
                mask_phase_deg=float(row["mask_phase"]),
                angular_difference_deg=float(angular_difference),
            )
        )
    return entries


def quantize_angle(angle_deg: float, increment_deg: float) -> float:
    if increment_deg <= 0.0:
        raise ValueError("the angular increment must be positive")
    return round(float(angle_deg) / float(increment_deg)) * float(increment_deg)


def _cosine_pattern(
    fx: float,
    fy: float,
    phase_rad: float,
    size: int,
) -> np.ndarray:
    axis = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    x, y = np.meshgrid(axis, axis, indexing="xy")
    return (
        0.5 + 0.5 * np.cos(2.0 * np.pi * (fx * x + fy * y) / size + phase_rad)
    ).astype(np.float32)


def corrected_mask_frequency(entry: AngleLibraryEntry) -> float:
    """Preserve the target sum-vector magnitude after introducing an angle split."""
    theta1 = math.radians(entry.target_angle_deg - entry.angular_difference_deg)
    theta2 = math.radians(entry.target_angle_deg + entry.angular_difference_deg)
    target_frequency = math.hypot(entry.target_fx, entry.target_fy)
    a = 1.0
    b = 2.0 * entry.slm_frequency * math.cos(theta1 - theta2)
    c = entry.slm_frequency**2 - target_frequency**2
    discriminant = max(b * b - 4.0 * a * c, 0.0)
    roots = ((-b + math.sqrt(discriminant)) / 2.0, (-b - math.sqrt(discriminant)) / 2.0)
    positive = [root for root in roots if root > 0.0]
    if not positive:
        raise ValueError(f"no positive mask-frequency solution for target {entry.point}")
    return round(min(positive, key=lambda root: abs(root - entry.mask_frequency)), 3)


def corrected_base_angle(entry: AngleLibraryEntry, mask_frequency: float) -> float:
    denominator = entry.slm_frequency + mask_frequency
    correction = 0.0 if denominator == 0.0 else (
        (entry.slm_frequency - mask_frequency) / denominator
    )
    return round(
        entry.target_angle_deg
        + correction * entry.angular_difference_deg * 1.003,
        3,
    )


def with_quantized_orientation(
    entry: AngleLibraryEntry,
    increment_deg: float,
) -> AngleLibraryEntry:
    """Quantize the mechanically commanded mean grating orientation."""
    frequency = corrected_mask_frequency(entry) if entry.slm_frequency else entry.mask_frequency
    angle = corrected_base_angle(entry, frequency) if entry.slm_frequency else entry.target_angle_deg
    return replace(entry, target_angle_deg=quantize_angle(angle, increment_deg))


def equivalent_illumination_pattern(
    entry: AngleLibraryEntry,
    *,
    size: int,
    slm_pixels: int,
    coefficients: tuple[float, float, float, float, float] = DEFAULT_EQ10_COEFFICIENTS,
    angle_increment_deg: float | None = None,
) -> np.ndarray:
    """Generate the nonnegative equivalent illumination used by the ideal study.

    The SLM-domain term is sampled on ``slm_pixels`` and expanded by exact
    nearest-neighbor pixel replication. The external carrier remains defined on
    the full grid. No bicubic interpolation is used.
    """
    if size % slm_pixels:
        raise ValueError("size must be an integer multiple of slm_pixels")
    factor = size // slm_pixels
    current = (
        with_quantized_orientation(entry, angle_increment_deg)
        if angle_increment_deg is not None
        else entry
    )
    total_phase = math.radians(current.total_phase_deg)
    if current.slm_frequency == 0.0:
        return _cosine_pattern(current.target_fx, current.target_fy, total_phase, size)

    mask_frequency = corrected_mask_frequency(current)
    base_angle = corrected_base_angle(current, mask_frequency)
    if angle_increment_deg is not None:
        base_angle = quantize_angle(base_angle, angle_increment_deg)
    theta_external = math.radians(base_angle - current.angular_difference_deg)
    theta_slm = math.radians(base_angle + current.angular_difference_deg)
    external = _cosine_pattern(
        current.slm_frequency * math.cos(theta_external),
        current.slm_frequency * math.sin(theta_external),
        math.radians(current.slm_phase_deg),
        size,
    )
    slm_low = _cosine_pattern(
        mask_frequency * math.cos(theta_slm),
        mask_frequency * math.sin(theta_slm),
        math.radians(current.mask_phase_deg),
        slm_pixels,
    )
    slm = np.repeat(np.repeat(slm_low, factor, axis=0), factor, axis=1)
    mixed = external * slm

    sum_fx = current.slm_frequency * math.cos(theta_external) + mask_frequency * math.cos(theta_slm)
    sum_fy = current.slm_frequency * math.sin(theta_external) + mask_frequency * math.sin(theta_slm)
    target = _cosine_pattern(sum_fx, sum_fy, total_phase, size)
    residual = np.clip(
        4.0 * (mixed + 0.5 - 0.5 * external - 0.5 * slm - 0.25 * target),
        0.0,
        1.0,
    )
    residual_low = residual.reshape(
        slm_pixels, factor, slm_pixels, factor
    ).mean(axis=(1, 3))
    residual_slm = np.repeat(np.repeat(residual_low, factor, axis=0), factor, axis=1)
    weights = np.asarray(coefficients, dtype=np.float32)
    result = (
        weights[0] * mixed
        + weights[1] * external
        + weights[2] * slm
        + weights[3] * residual_slm
        + weights[4]
    )
    return np.clip(result, 0.0, 1.0).astype(np.float32)
