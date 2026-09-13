from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import sparse

from ..metrics import image_metrics, normalize_image
from .config import PhysicalSLMConfig as PhysicalBandwidthConfig
from .order_model import (
    target_intensity,
)
from .pattern_cache import PhysicalQuartetCache


def half_plane_frequency_points(radius: float) -> list[tuple[int, int]]:
    limit = int(math.floor(radius))
    points = []
    for qy in range(-limit, limit + 1):
        for qx in range(-limit, limit + 1):
            if qx == 0 and qy == 0:
                continue
            if not (qy > 0 or (qy == 0 and qx > 0)):
                continue
            if qx * qx + qy * qy <= radius * radius + 1e-12:
                points.append((qx, qy))
    return sorted(points, key=lambda q: (q[0] * q[0] + q[1] * q[1], q[1], q[0]))


def parallel_frequency_decomposition(target: np.ndarray, fmax: float) -> dict[str, np.ndarray]:
    vector = np.asarray(target, dtype=np.float64)
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= fmax + 1e-12:
        mask = np.zeros(2, dtype=np.float64)
        slm = vector.copy()
    else:
        if magnitude > 2.0 * fmax + 1e-12:
            raise ValueError("target frequency exceeds the nominal MSF radius")
        mask = fmax * vector / magnitude
        slm = vector - mask
    return {
        "target": vector,
        "slm": slm,
        "mask": mask,
        "difference": slm - mask,
    }


def four_step_coefficient(buckets: np.ndarray) -> complex:
    values = np.asarray(buckets, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError("four-step demodulation requires four bucket values")
    return complex(values[0] - values[2], values[1] - values[3])


def four_step_kernel(patterns: np.ndarray) -> np.ndarray:
    values = np.asarray(patterns, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 4:
        raise ValueError("four-step patterns must have shape (4, y, x)")
    return (values[0] - values[2]) + 1j * (values[1] - values[3])


def calibrate_demodulated_coefficient(
    raw_coefficient: complex,
    measured_kernel: np.ndarray,
    qx: int,
    qy: int,
    white_bucket: float,
) -> complex:
    """Correct a four-step coefficient using an object-independent system kernel.

    The measured kernel is decomposed into a DC term, the desired complex
    sinusoid, and its conjugate.  The DC response is removed with the one-time
    all-pass bucket measurement and the resulting 2x2 conjugate system is
    inverted.  Residual orthogonal pattern distortion is deliberately retained.
    """
    kernel = np.asarray(measured_kernel, dtype=np.complex128)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("measured kernel must be a square two-dimensional array")
    size = kernel.shape[0]
    axis = np.arange(size, dtype=np.float64) + 0.5 - size / 2.0
    x, y = np.meshgrid(axis, axis, indexing="xy")
    target = np.exp(-2j * np.pi * (qx * x + qy * y) / size)
    dc_response = complex(kernel.mean())
    zero_mean = kernel - dc_response
    sample_count = float(kernel.size)
    gain = np.vdot(target, zero_mean) / sample_count
    conjugate_gain = np.vdot(np.conj(target), zero_mean) / sample_count
    denominator = abs(gain) ** 2 - abs(conjugate_gain) ** 2
    if abs(denominator) < 1e-12:
        raise ValueError("four-step system response is singular at this frequency")
    corrected = complex(raw_coefficient) - dc_response * float(white_bucket)
    return (
        np.conj(gain) * corrected - conjugate_gain * np.conj(corrected)
    ) / denominator


def centered_to_fft_coefficient(value: complex, qx: int, qy: int, size: int) -> complex:
    offset = 0.5 - size / 2.0
    return value * np.exp(2j * np.pi * (qx + qy) * offset / size)


def _vector_polar(vector: np.ndarray) -> tuple[float, float]:
    frequency = float(np.linalg.norm(vector))
    angle = math.degrees(math.atan2(float(vector[1]), float(vector[0]))) if frequency else 0.0
    return frequency, angle


def _ideal_mask(size: int, vector: np.ndarray, phase_deg: float = 0.0) -> np.ndarray:
    frequency, angle = _vector_polar(vector)
    return target_intensity(size, frequency, angle, phase_deg)


def _bandlimited_reference(image: np.ndarray, radius: float) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    fy = np.fft.fftshift(np.fft.fftfreq(values.shape[0]) * values.shape[0])
    fx = np.fft.fftshift(np.fft.fftfreq(values.shape[1]) * values.shape[1])
    qx, qy = np.meshgrid(fx, fy, indexing="xy")
    support = qx**2 + qy**2 <= radius**2 + 1e-12
    spectrum = np.fft.fftshift(np.fft.fft2(values)) * support
    return normalize_image(np.real(np.fft.ifft2(np.fft.ifftshift(spectrum))))


def _reconstruct(spectrum: np.ndarray) -> np.ndarray:
    return normalize_image(np.real(np.fft.ifft2(spectrum)))


def run_fmax_comparison(
    image: np.ndarray,
    cfg: PhysicalBandwidthConfig,
    *,
    fmax: float,
    carrier: float,
    iris_mm: float,
    cache_dir: Path,
    response_matrix_path: Path | None = None,
    rebuild_response: bool = False,
    progress_every: int = 25,
) -> dict:
    # Local imports avoid a module cycle: the shared response module also uses
    # the coordinate conversion defined in this file.
    from ..calibration.sparse_response import (
        CalibratedResponseSolver,
        apply_real_linear_matrix,
        matrix_from_kernels,
        response_from_kernel,
    )

    reference = normalize_image(image)
    size = cfg.slm_pixels * cfg.oversample
    if reference.shape != (size, size):
        raise ValueError(f"reference image must have shape {(size, size)}")
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    pattern_dir = cache_root / "patterns"
    pattern_dir.mkdir(parents=True, exist_ok=True)
    quartet_cache = PhysicalQuartetCache(
        cfg, carrier=carrier, iris_mm=iris_mm, root=pattern_dir
    )

    low_points = half_plane_frequency_points(fmax)
    all_points = half_plane_frequency_points(2.0 * fmax)
    high_points = [q for q in all_points if q[0] * q[0] + q[1] * q[1] > fmax * fmax]
    spi_spectrum_raw = np.zeros((size, size), dtype=np.complex128)
    msf_spectrum_raw = np.zeros_like(spi_spectrum_raw)
    spi_spectrum = np.zeros_like(spi_spectrum_raw)
    msf_spectrum = np.zeros_like(spi_spectrum_raw)
    dc = float(reference.sum())
    for spectrum in (spi_spectrum_raw, msf_spectrum_raw, spi_spectrum, msf_spectrum):
        spectrum[0, 0] = dc
    direct_exposures = 0
    calibrated_coefficients: list[complex] = []
    nominal_responses = []

    def store(spectrum: np.ndarray, qx: int, qy: int, coefficient: complex) -> None:
        converted = centered_to_fft_coefficient(coefficient, qx, qy, size)
        spectrum[qy % size, qx % size] = converted
        spectrum[-qy % size, -qx % size] = np.conj(converted)

    for index, (qx, qy) in enumerate(low_points, start=1):
        patterns = quartet_cache.get(np.asarray((qx, qy), dtype=float))
        buckets = np.sum(patterns * reference[None, :, :], axis=(1, 2), dtype=np.float64)
        raw_coefficient = four_step_coefficient(buckets)
        measured_kernel = four_step_kernel(patterns)
        calibrated_coefficient = calibrate_demodulated_coefficient(
            raw_coefficient, measured_kernel, qx, qy, dc
        )
        calibrated_coefficients.append(calibrated_coefficient)
        nominal_responses.append(response_from_kernel(measured_kernel, qx, qy))
        store(spi_spectrum_raw, qx, qy, raw_coefficient)
        store(msf_spectrum_raw, qx, qy, raw_coefficient)
        store(spi_spectrum, qx, qy, calibrated_coefficient)
        store(msf_spectrum, qx, qy, calibrated_coefficient)
        direct_exposures += 4
        if progress_every and index % progress_every == 0:
            print(f"  direct SPI frequencies: {index}/{len(low_points)}", flush=True)

    white_bucket = dc
    mask_keys: set[tuple[float, float]] = set()
    moire_exposures = 0
    for index, (qx, qy) in enumerate(high_points, start=1):
        decomposition = parallel_frequency_decomposition(
            np.asarray((qx, qy), dtype=float), fmax
        )
        slm_patterns = quartet_cache.get(decomposition["slm"])
        difference_patterns = quartet_cache.get(decomposition["difference"])
        mask = _ideal_mask(size, decomposition["mask"])
        mask_bucket = float(np.sum(mask * reference, dtype=np.float64))
        mask_keys.add(tuple(np.round(decomposition["mask"], 8)))
        equivalent_buckets = np.empty(4, dtype=np.float64)
        for phase_index in range(4):
            slm_bucket = float(
                np.sum(slm_patterns[phase_index] * reference, dtype=np.float64)
            )
            difference_bucket = float(
                np.sum(difference_patterns[phase_index] * reference, dtype=np.float64)
            )
            moire_bucket = float(
                np.sum(slm_patterns[phase_index] * mask * reference, dtype=np.float64)
            )
            equivalent_buckets[phase_index] = (
                4.0 * moire_bucket
                - 2.0 * slm_bucket
                - 2.0 * mask_bucket
                - difference_bucket
                + 2.0 * white_bucket
            )
        raw_coefficient = four_step_coefficient(equivalent_buckets)
        slm_kernel = four_step_kernel(slm_patterns)
        difference_kernel = four_step_kernel(difference_patterns)
        equivalent_kernel = (4.0 * mask - 2.0) * slm_kernel - difference_kernel
        calibrated_coefficient = calibrate_demodulated_coefficient(
            raw_coefficient, equivalent_kernel, qx, qy, dc
        )
        calibrated_coefficients.append(calibrated_coefficient)
        nominal_responses.append(response_from_kernel(equivalent_kernel, qx, qy))
        store(msf_spectrum_raw, qx, qy, raw_coefficient)
        store(msf_spectrum, qx, qy, calibrated_coefficient)
        moire_exposures += 4
        if progress_every and index % progress_every == 0:
            print(f"  MSF high frequencies: {index}/{len(high_points)}", flush=True)

    unique_quartet_exposures = 4 * quartet_cache.unique_vector_count
    unique_exposures = 1 + unique_quartet_exposures + len(mask_keys) + moire_exposures
    naive_exposures = 1 + direct_exposures + 13 * len(high_points)
    spi_reconstruction_raw = _reconstruct(spi_spectrum_raw)
    msf_reconstruction_raw = _reconstruct(msf_spectrum_raw)
    spi_reconstruction = _reconstruct(spi_spectrum)
    msf_reconstruction = _reconstruct(msf_spectrum)
    ideal_spi = _bandlimited_reference(reference, fmax)
    ideal_msf = _bandlimited_reference(reference, 2.0 * fmax)

    def nominal_kernels():
        for qx, qy in all_points:
            target = np.asarray((qx, qy), dtype=float)
            if qx * qx + qy * qy <= fmax * fmax:
                yield four_step_kernel(quartet_cache.get(target))
            else:
                decomposition = parallel_frequency_decomposition(target, fmax)
                slm_patterns = quartet_cache.get(decomposition["slm"])
                difference_patterns = quartet_cache.get(decomposition["difference"])
                mask = _ideal_mask(size, decomposition["mask"])
                yield (4.0 * mask - 2.0) * four_step_kernel(slm_patterns) - four_step_kernel(
                    difference_patterns
                )

    matrix_path = (
        Path(response_matrix_path).resolve()
        if response_matrix_path is not None
        else cache_root / "nominal_sparse_response_matrix.npz"
    )
    if matrix_path.exists() and not rebuild_response:
        response_matrix = sparse.load_npz(matrix_path)
        expected_shape = (2 * len(all_points), 2 * len(all_points))
        if response_matrix.shape != expected_shape:
            raise ValueError(
                f"cached sparse response matrix has shape {response_matrix.shape}, "
                f"expected {expected_shape}"
            )
    else:
        response_matrix = matrix_from_kernels(
            nominal_kernels(),
            all_points,
            nominal_responses,
            max_terms=96,
            relative_threshold=5e-4,
        )
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(matrix_path, response_matrix)
    measured_coefficients = np.asarray(calibrated_coefficients, dtype=np.complex128)
    response_solver = CalibratedResponseSolver(response_matrix)
    sparse_coefficients = response_solver.solve(measured_coefficients)
    predicted_coefficients = apply_real_linear_matrix(response_matrix, sparse_coefficients)
    sparse_residual = float(
        np.linalg.norm(predicted_coefficients - measured_coefficients)
        / max(float(np.linalg.norm(measured_coefficients)), 1e-20)
    )
    msf_sparse_spectrum = np.zeros_like(msf_spectrum)
    msf_sparse_spectrum[0, 0] = dc
    for (qx, qy), coefficient in zip(all_points, sparse_coefficients):
        store(msf_sparse_spectrum, qx, qy, coefficient)
    msf_sparse_reconstruction = _reconstruct(msf_sparse_spectrum)
    result = {
        "reference": reference,
        "spi_reconstruction": spi_reconstruction,
        "msf_reconstruction": msf_reconstruction,
        "msf_sparse_reconstruction": msf_sparse_reconstruction,
        "spi_reconstruction_raw": spi_reconstruction_raw,
        "msf_reconstruction_raw": msf_reconstruction_raw,
        "ideal_spi_reference": ideal_spi,
        "ideal_msf_reference": ideal_msf,
        "spi_spectrum": spi_spectrum,
        "msf_spectrum": msf_spectrum,
        "msf_sparse_spectrum": msf_sparse_spectrum,
        "spi_spectrum_raw": spi_spectrum_raw,
        "msf_spectrum_raw": msf_spectrum_raw,
        "spi_metrics": image_metrics(reference, spi_reconstruction),
        "msf_metrics": image_metrics(reference, msf_reconstruction),
        "msf_sparse_metrics": image_metrics(reference, msf_sparse_reconstruction),
        "spi_raw_metrics": image_metrics(reference, spi_reconstruction_raw),
        "msf_raw_metrics": image_metrics(reference, msf_reconstruction_raw),
        "spi_vs_ideal_metrics": image_metrics(ideal_spi, spi_reconstruction),
        "msf_vs_ideal_metrics": image_metrics(ideal_msf, msf_reconstruction),
        "msf_sparse_vs_ideal_metrics": image_metrics(
            ideal_msf, msf_sparse_reconstruction
        ),
        "spi_frequency_count": 1 + 2 * len(low_points),
        "msf_frequency_count": 1 + 2 * len(all_points),
        "low_half_plane_count": len(low_points),
        "high_half_plane_count": len(high_points),
        "unique_pattern_quartets": quartet_cache.unique_vector_count,
        "unique_mask_angles": len(mask_keys),
        "unique_exposure_count": unique_exposures,
        "naive_exposure_count": naive_exposures,
        "fmax": float(fmax),
        "carrier": float(carrier),
        "iris_mm": float(iris_mm),
        "fill_factor": float(cfg.fill_factor),
        "calibration_method": (
            "per-frequency DC/gain/conjugate preconditioning followed by an "
            "object-independent sparse real-linear response solve"
        ),
        "sparse_response": {
            "matrix_path": str(matrix_path),
            "shape": list(response_matrix.shape),
            "nonzeros": int(response_matrix.nnz),
            "max_terms_per_complex_row": 96,
            "relative_threshold": 5e-4,
            "relative_forward_residual": sparse_residual,
        },
    }
    np.savez_compressed(
        cache_root / "comparison_cache.npz",
        reference=reference,
        spi_reconstruction=spi_reconstruction,
        msf_reconstruction=msf_reconstruction,
        msf_sparse_reconstruction=msf_sparse_reconstruction,
        spi_reconstruction_raw=spi_reconstruction_raw,
        msf_reconstruction_raw=msf_reconstruction_raw,
        ideal_spi_reference=ideal_spi,
        ideal_msf_reference=ideal_msf,
        spi_spectrum=spi_spectrum,
        msf_spectrum=msf_spectrum,
        msf_sparse_spectrum=msf_sparse_spectrum,
        spi_spectrum_raw=spi_spectrum_raw,
        msf_spectrum_raw=msf_spectrum_raw,
    )
    return result
