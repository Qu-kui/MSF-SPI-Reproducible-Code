from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def half_plane_frequency_points(radius: float) -> list[tuple[int, int]]:
    """Enumerate one member of each nonzero Hermitian Fourier pair."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    limit = int(math.floor(radius))
    points: list[tuple[int, int]] = []
    for qy in range(-limit, limit + 1):
        for qx in range(-limit, limit + 1):
            if qx == 0 and qy == 0:
                continue
            if not (qy > 0 or (qy == 0 and qx > 0)):
                continue
            if qx * qx + qy * qy <= radius * radius + 1e-12:
                points.append((qx, qy))
    return sorted(points, key=lambda q: (q[0] * q[0] + q[1] * q[1], q[1], q[0]))


def four_step_coefficient(buckets: np.ndarray) -> complex:
    """Demodulate four bucket measurements ordered at 0, 90, 180, 270 degrees."""
    values = np.asarray(buckets, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError("four-step demodulation requires four bucket values")
    return complex(values[0] - values[2], values[1] - values[3])


def centered_to_fft_coefficient(value: complex, qx: int, qy: int, size: int) -> complex:
    """Apply the half-pixel phase convention used by the centered pattern grid."""
    offset = 0.5 - size / 2.0
    return complex(value) * np.exp(2j * np.pi * (qx + qy) * offset / size)


def spectrum_from_half_plane(
    coefficients: Mapping[tuple[int, int], complex],
    *,
    size: int,
    dc: float | complex = 0.0,
) -> np.ndarray:
    """Build a Hermitian FFT array from unique half-plane coefficients."""
    if size < 2:
        raise ValueError("size must be at least two")
    spectrum = np.zeros((size, size), dtype=np.complex128)
    spectrum[0, 0] = complex(dc) * size * size
    for (qx, qy), value in coefficients.items():
        if qx == 0 and qy == 0:
            raise ValueError("supply the zero-frequency term through dc")
        coefficient = complex(value)
        spectrum[qy % size, qx % size] = coefficient
        spectrum[-qy % size, -qx % size] = np.conj(coefficient)
    return spectrum


def reconstruct_from_half_plane(
    coefficients: Mapping[tuple[int, int], complex],
    *,
    size: int,
    dc: float | complex = 0.0,
) -> np.ndarray:
    """Reconstruct a real image from half-plane Fourier coefficients."""
    spectrum = spectrum_from_half_plane(coefficients, size=size, dc=dc)
    return np.fft.ifft2(spectrum).real

