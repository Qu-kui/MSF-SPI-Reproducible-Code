from __future__ import annotations

import math

import numpy as np
from skimage.metrics import structural_similarity


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize an image to [0, 1], returning zeros for a constant image."""
    values = np.asarray(image, dtype=float)
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def image_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Return SSIM, PSNR, and normalized root-mean-square error."""
    ref = normalize_image(reference)
    img = normalize_image(candidate)
    difference = img - ref
    mse = float(np.mean(difference**2))
    psnr = math.inf if mse <= 1e-15 else float(10.0 * math.log10(1.0 / mse))
    denominator = max(float(np.linalg.norm(ref)), 1e-15)
    return {
        "ssim": float(structural_similarity(ref, img, data_range=1.0)),
        "psnr_db": psnr,
        "nrmse": float(np.linalg.norm(difference) / denominator),
    }


def spectral_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    high_band_mask: np.ndarray,
) -> dict[str, float]:
    """Compare complex spectra over a selected high-frequency support."""
    ref = np.asarray(reference, dtype=complex)[high_band_mask]
    cur = np.asarray(candidate, dtype=complex)[high_band_mask]
    denominator = float(np.linalg.norm(ref) * np.linalg.norm(cur))
    correlation = float(abs(np.vdot(ref, cur)) / denominator) if denominator else 0.0
    ref_energy = float(np.sum(np.abs(ref) ** 2))
    cur_energy = float(np.sum(np.abs(cur) ** 2))
    nrmse = float(np.linalg.norm(cur - ref) / max(np.linalg.norm(ref), 1e-15))
    return {
        "spectral_correlation_high": correlation,
        "spectral_nrmse_high": nrmse,
        "high_band_energy_recovery": cur_energy / ref_energy if ref_energy else 0.0,
    }

