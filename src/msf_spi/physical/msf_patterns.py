from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import PhysicalSLMConfig as PhysicalBandwidthConfig
from .order_model import (
    normalize_phase_sequence,
    simulate_recovered_pattern,
    sinusoid_quality,
    target_intensity,
)


@dataclass(frozen=True)
class FrequencyDecomposition:
    target_vector: np.ndarray
    slm_vector: np.ndarray
    mask_vector: np.ndarray


def circular_frequency_points(radius: float, step: float = 1.0) -> list[np.ndarray]:
    limit = int(math.ceil(radius / step))
    points = []
    for iy in range(-limit, limit + 1):
        for ix in range(-limit, limit + 1):
            point = np.asarray((ix * step, iy * step), dtype=np.float64)
            if np.linalg.norm(point) <= radius + 1e-12:
                points.append(point)
    return points


def decompose_target_frequency(
    target_vector: np.ndarray, slm_radius: float
) -> FrequencyDecomposition:
    target = np.asarray(target_vector, dtype=np.float64)
    magnitude = float(np.linalg.norm(target))
    if magnitude <= slm_radius or magnitude == 0.0:
        mask = np.zeros(2, dtype=np.float64)
        slm = target.copy()
    else:
        if magnitude > 2.0 * slm_radius + 1e-12:
            raise ValueError("target lies outside the two-carrier sum-frequency disk")
        direction = target / magnitude
        perpendicular = np.asarray((-direction[1], direction[0]))
        half_angle = math.acos(np.clip(magnitude / (2.0 * slm_radius), -1.0, 1.0))
        slm = slm_radius * (
            math.cos(half_angle) * direction + math.sin(half_angle) * perpendicular
        )
        mask = slm_radius * (
            math.cos(half_angle) * direction - math.sin(half_angle) * perpendicular
        )
    return FrequencyDecomposition(target, slm, mask)


def ideal_msf_equivalent_pattern(
    slm_intensity: np.ndarray,
    mask_intensity: np.ndarray,
    difference_intensity: np.ndarray,
    white_intensity: np.ndarray | float = 1.0,
) -> np.ndarray:
    slm = np.asarray(slm_intensity, dtype=np.float64)
    mask = np.asarray(mask_intensity, dtype=np.float64)
    difference = np.asarray(difference_intensity, dtype=np.float64)
    white = np.asarray(white_intensity, dtype=np.float64)
    return 4.0 * slm * mask - 2.0 * slm - 2.0 * mask - difference + 2.0 * white


def physical_msf_phase_sequence(
    cfg: PhysicalBandwidthConfig,
    target_frequency: float,
    angle_deg: float,
    slm_bandwidth: float,
    carrier_frequency: float,
    iris_radius_mm: float,
) -> dict:
    phases = (0.0, 90.0, 180.0, 270.0)
    theta = math.radians(angle_deg)
    target_vector = target_frequency * np.asarray((math.cos(theta), math.sin(theta)))
    decomposition = decompose_target_frequency(target_vector, slm_bandwidth)
    slm_frequency = float(np.linalg.norm(decomposition.slm_vector))
    slm_angle = math.degrees(
        math.atan2(decomposition.slm_vector[1], decomposition.slm_vector[0])
    )
    simulations = [
        simulate_recovered_pattern(
            slm_frequency,
            slm_angle,
            phase,
            carrier_frequency,
            iris_radius_mm,
            cfg,
        )
        for phase in phases
    ]
    slm_patterns = normalize_phase_sequence(
        np.stack([item.recovered_intensity for item in simulations])
    )
    size = cfg.slm_pixels * cfg.oversample
    if target_frequency <= slm_bandwidth:
        equivalent = slm_patterns
    else:
        mask_frequency = float(np.linalg.norm(decomposition.mask_vector))
        mask_angle = math.degrees(
            math.atan2(decomposition.mask_vector[1], decomposition.mask_vector[0])
        )
        difference_vector = decomposition.slm_vector - decomposition.mask_vector
        difference_frequency = float(np.linalg.norm(difference_vector))
        difference_angle = math.degrees(
            math.atan2(difference_vector[1], difference_vector[0])
        )
        mask = target_intensity(size, mask_frequency, mask_angle, 0.0)
        equivalent = np.stack(
            [
                ideal_msf_equivalent_pattern(
                    slm_patterns[index],
                    mask,
                    target_intensity(
                        size,
                        difference_frequency,
                        difference_angle,
                        phase,
                    ),
                )
                for index, phase in enumerate(phases)
            ]
        )
    targets = np.stack(
        [target_intensity(size, target_frequency, angle_deg, phase) for phase in phases]
    )
    qualities = [
        sinusoid_quality(
            targets[index],
            equivalent[index],
            target_frequency,
            angle_deg,
            phase,
            cfg.thresholds,
        )
        for index, phase in enumerate(phases)
    ]
    return {
        "targets": targets,
        "slm_patterns": slm_patterns,
        "equivalent_patterns": equivalent,
        "qualities": qualities,
        "slm_frequency": float(slm_frequency),
        "mask_frequency": float(np.linalg.norm(decomposition.mask_vector)),
    }
