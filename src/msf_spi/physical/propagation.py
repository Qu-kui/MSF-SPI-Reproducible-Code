from __future__ import annotations

import numpy as np


def _inverse_sinc_nonnegative(amplitude: np.ndarray, iterations: int = 64) -> np.ndarray:
    """Invert sinc(q) over q in [0, pi] for nonnegative normalized amplitudes."""
    target = np.clip(np.asarray(amplitude, dtype=np.float64), 0.0, 1.0)
    low = np.zeros_like(target)
    high = np.full_like(target, np.pi)
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        value = np.sinc(mid / np.pi)
        move_low = value > target
        low = np.where(move_low, mid, low)
        high = np.where(move_low, high, mid)
    return 0.5 * (low + high)


def gaussian_amplitude(size: int, pixel_pitch: float, radius: float) -> np.ndarray:
    """Return the sampled amplitude of a circular Gaussian beam."""
    axis = (np.arange(size, dtype=np.float64) + 0.5 - size / 2.0) * pixel_pitch
    x, y = np.meshgrid(axis, axis, indexing="xy")
    return np.exp(-(x**2 + y**2) / radius**2)

