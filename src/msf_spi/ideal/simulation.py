from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from ..calibration.sparse_response import (
    apply_frozen_response,
    four_step_kernel,
    matrix_from_kernels,
    response_from_kernel,
    solve_calibrated_coefficients,
)
from ..metrics import image_metrics, normalize_image
from ..reconstruction.fourier import centered_to_fft_coefficient
from .patterns import AngleLibraryEntry, equivalent_illumination_pattern


PHASES_DEG = (0, 90, 180, 270)


def _phase_index(entry: AngleLibraryEntry) -> int:
    phase = int(round(entry.total_phase_deg)) % 360
    distances = [min(abs(phase - wanted), 360 - abs(phase - wanted)) for wanted in PHASES_DEG]
    index = int(np.argmin(distances))
    if distances[index] > 1:
        raise ValueError(f"unsupported total phase: {entry.total_phase_deg}")
    return index


def select_frequency_quartets(
    entries: Iterable[AngleLibraryEntry],
    *,
    maximum_radius: float | None = None,
    max_points: int | None = None,
) -> dict[tuple[int, int], tuple[AngleLibraryEntry, ...]]:
    grouped: dict[tuple[int, int], list[AngleLibraryEntry | None]] = defaultdict(
        lambda: [None, None, None, None]
    )
    for entry in entries:
        qx, qy = entry.point
        if not (qy > 0 or (qy == 0 and qx > 0)):
            continue
        if maximum_radius is not None and math_hypot(qx, qy) > maximum_radius + 1e-12:
            continue
        slot = _phase_index(entry)
        if grouped[(qx, qy)][slot] is not None:
            raise ValueError(f"duplicate phase at frequency {(qx, qy)}")
        grouped[(qx, qy)][slot] = entry
    points = sorted(grouped, key=lambda q: (q[0] ** 2 + q[1] ** 2, q[1], q[0]))
    if max_points is not None and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = [points[index] for index in np.unique(indices)]
    result: dict[tuple[int, int], tuple[AngleLibraryEntry, ...]] = {}
    for point in points:
        values = grouped[point]
        if any(value is None for value in values):
            raise ValueError(f"frequency {point} does not have four phase entries")
        result[point] = tuple(values)  # type: ignore[arg-type]
    return result


def math_hypot(x: float, y: float) -> float:
    return float(np.hypot(float(x), float(y)))


def _reconstruct(
    size: int,
    points: list[tuple[int, int]],
    coefficients: np.ndarray,
    dc_bucket: float,
) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.zeros((size, size), dtype=np.complex128)
    spectrum[0, 0] = dc_bucket
    for (qx, qy), value in zip(points, coefficients):
        converted = centered_to_fft_coefficient(value, qx, qy, size)
        spectrum[qy % size, qx % size] = converted
        spectrum[-qy % size, -qx % size] = np.conj(converted)
    return normalize_image(np.fft.ifft2(spectrum).real), spectrum


def run_ideal_reconstruction(
    image: np.ndarray,
    entries: Iterable[AngleLibraryEntry],
    *,
    slm_pixels: int,
    maximum_radius: float,
    angle_increment_deg: float = 0.001,
    coefficients: tuple[float, float, float, float, float] = (
        2.2326,
        -1.1163,
        -1.1163,
        -0.5928,
        1.3546,
    ),
    regularization: float = 0.1,
    max_points: int | None = None,
) -> dict:
    """Acquire, calibrate, and reconstruct the ideal pixel-replicated model."""
    object_image = normalize_image(np.asarray(image, dtype=np.float64))
    if object_image.ndim != 2 or object_image.shape[0] != object_image.shape[1]:
        raise ValueError("image must be a square two-dimensional array")
    size = object_image.shape[0]
    groups = select_frequency_quartets(
        entries,
        maximum_radius=maximum_radius,
        max_points=max_points,
    )
    if not groups:
        raise ValueError("the selected angle-library subset is empty")
    points = list(groups)
    quartets: list[np.ndarray] = []
    buckets = np.empty((len(points), 4), dtype=np.float64)
    for index, point in enumerate(points):
        patterns = np.stack(
            [
                equivalent_illumination_pattern(
                    entry,
                    size=size,
                    slm_pixels=slm_pixels,
                    coefficients=coefficients,
                    angle_increment_deg=angle_increment_deg,
                )
                for entry in groups[point]
            ]
        )
        quartets.append(patterns)
        buckets[index] = np.einsum("kij,ij->k", patterns, object_image)
    dc_bucket = float(object_image.sum())
    raw = (buckets[:, 0] - buckets[:, 2]) + 1j * (buckets[:, 1] - buckets[:, 3])
    kernels = [four_step_kernel(patterns) for patterns in quartets]
    responses = [
        response_from_kernel(kernel, *point) for kernel, point in zip(kernels, points)
    ]
    corrected = np.asarray(
        [
            apply_frozen_response(value, response, dc_bucket)
            for value, response in zip(raw, responses)
        ]
    )
    response_matrix = matrix_from_kernels(kernels, points, responses)
    calibrated = solve_calibrated_coefficients(
        response_matrix,
        corrected,
        regularization=regularization,
    )
    direct_image, direct_spectrum = _reconstruct(size, points, corrected, dc_bucket)
    calibrated_image, calibrated_spectrum = _reconstruct(
        size, points, calibrated, dc_bucket
    )
    return {
        "points": points,
        "frequency_points": len(points),
        "phase_measurements": 4 * len(points),
        "angle_increment_deg": float(angle_increment_deg),
        "direct_reconstruction": direct_image,
        "sparse_reconstruction": calibrated_image,
        "direct_spectrum": direct_spectrum,
        "sparse_spectrum": calibrated_spectrum,
        "response_matrix": response_matrix,
        "direct_metrics": image_metrics(object_image, direct_image),
        "sparse_metrics": image_metrics(object_image, calibrated_image),
    }
