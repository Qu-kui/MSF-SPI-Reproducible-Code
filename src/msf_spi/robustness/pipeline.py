from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse

from ..metrics import image_metrics, normalize_image
from .errors import calibration_drift
from ..physical.config import PhysicalSLMConfig as PhysicalBandwidthConfig
from ..physical.comparison import (
    _bandlimited_reference,
    half_plane_frequency_points,
    parallel_frequency_decomposition,
)
from ..calibration.global_weights import fit_global_weights
from .cases import (
    FrozenResponse,
    PHASES_DEG,
    ReducedOrderModel,
    RobustnessCase,
    apply_frozen_response,
    build_robustness_cases,
    coefficients_to_reconstruction,
    four_step_coefficients,
    four_step_kernel,
    response_from_kernel,
    zero_mean_rms_map,
)
from ..calibration.sparse_response import (
    CalibratedResponseSolver,
    matrix_from_kernels,
)


def calibration_vectors(fmax: float) -> list[np.ndarray]:
    radii = np.asarray((1.15, 1.40, 1.70, 1.95)) * float(fmax)
    angles = np.deg2rad(np.arange(0.0, 180.0, 15.0))
    return [
        np.asarray((radius * np.cos(theta), radius * np.sin(theta)), dtype=np.float64)
        for radius in radii
        for theta in angles
    ]


def calibration_validation_vectors(fmax: float) -> list[np.ndarray]:
    radii = np.asarray((1.25, 1.55, 1.85)) * float(fmax)
    angles = np.deg2rad(np.arange(7.5, 180.0, 15.0))
    return [
        np.asarray((radius * np.cos(theta), radius * np.sin(theta)), dtype=np.float64)
        for radius in radii
        for theta in angles
    ]


def _mask(model: ReducedOrderModel, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(vector, dtype=np.float64)
    return 0.5 + 0.5 * np.cos(2.0 * np.pi * (q[0] * model.x + q[1] * model.y))


def _downsample(values: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[-1] == size:
        return array
    leading = array.shape[:-2]
    output = np.empty((*leading, size, size), dtype=np.float32)
    for index in np.ndindex(leading):
        output[index] = cv2.resize(
            np.asarray(array[index], dtype=np.float32),
            (size, size),
            interpolation=cv2.INTER_AREA,
        )
    return output


def fit_reduced_global_weights(
    model: ReducedOrderModel,
    *,
    fmax: float,
    downsample: int = 64,
    iterations: int = 4,
) -> tuple[np.ndarray, dict]:
    records: list[tuple[np.ndarray, np.ndarray]] = []
    for target in calibration_vectors(fmax):
        decomposition = parallel_frequency_decomposition(target, fmax)
        slm = model.quartet(decomposition["slm"])
        difference = model.quartet(decomposition["difference"])
        mask = _mask(model, decomposition["mask"])
        target_quartet = model.target_intensities(target)
        features = np.stack(
            (
                slm * mask[None, :, :],
                slm,
                np.broadcast_to(mask, slm.shape),
                difference,
                np.ones_like(slm),
            ),
            axis=-1,
        )
        features_small = np.moveaxis(
            _downsample(np.moveaxis(features, -1, 0), downsample), 0, -1
        )
        records.append((features_small, _downsample(target_quartet, downsample)))
    weights, diagnostics = fit_global_weights(
        records, iterations=iterations, eps=1e-5, loss="l1"
    )
    return weights, {
        **diagnostics,
        "calibration_vector_count": len(records),
        "downsample": int(downsample),
    }


def evaluate_reduced_global_weights(
    model: ReducedOrderModel,
    weights: np.ndarray,
    *,
    fmax: float,
    downsample: int = 64,
) -> dict:
    theoretical = np.asarray((4.0, -2.0, -2.0, -1.0, 2.0), dtype=np.float64)
    fitted_sae = fitted_sse = baseline_sae = baseline_sse = 0.0
    count = 0
    for target in calibration_validation_vectors(fmax):
        decomposition = parallel_frequency_decomposition(target, fmax)
        slm = model.quartet(decomposition["slm"])
        difference = model.quartet(decomposition["difference"])
        mask = _mask(model, decomposition["mask"])
        features = np.stack(
            (
                slm * mask[None, :, :],
                slm,
                np.broadcast_to(mask, slm.shape),
                difference,
                np.ones_like(slm),
            ),
            axis=-1,
        )
        features = np.moveaxis(
            _downsample(np.moveaxis(features, -1, 0), downsample), 0, -1
        )
        target_values = _downsample(model.target_intensities(target), downsample)
        fitted_residual = np.tensordot(features, weights, axes=([-1], [0])) - target_values
        baseline_residual = np.tensordot(features, theoretical, axes=([-1], [0])) - target_values
        fitted_sae += float(np.sum(np.abs(fitted_residual)))
        fitted_sse += float(np.sum(fitted_residual**2))
        baseline_sae += float(np.sum(np.abs(baseline_residual)))
        baseline_sse += float(np.sum(baseline_residual**2))
        count += fitted_residual.size
    return {
        "validation_vector_count": len(calibration_validation_vectors(fmax)),
        "sample_count": int(count),
        "fitted_mae": fitted_sae / count,
        "fitted_mse": fitted_sse / count,
        "theoretical_mae": baseline_sae / count,
        "theoretical_mse": baseline_sse / count,
    }


def crosscheck_reduced_model(
    cfg: PhysicalBandwidthConfig,
    model: ReducedOrderModel,
    *,
    carrier: float,
    iris_mm: float,
) -> dict:
    from ..physical.order_model import normalize_phase_sequence, simulate_recovered_pattern

    rows = []
    for frequency, angle in ((28.0, 0.0), (28.0, 45.0), (32.0, 0.0), (32.0, 45.0)):
        theta = math.radians(angle)
        vector = np.asarray((frequency * math.cos(theta), frequency * math.sin(theta)))
        reduced = model.quartet(vector)
        explicit_items = [
            simulate_recovered_pattern(
                frequency, angle, phase, carrier, iris_mm, cfg
            )
            for phase in PHASES_DEG
        ]
        explicit = normalize_phase_sequence(
            np.stack([item.recovered_intensity for item in explicit_items])
        )
        factor = cfg.oversample
        explicit = explicit.reshape(
            4, cfg.slm_pixels, factor, cfg.slm_pixels, factor
        ).mean(axis=(2, 4))
        reduced_kernel = four_step_kernel(reduced).ravel()
        explicit_kernel = four_step_kernel(explicit).ravel()
        denominator = float(np.linalg.norm(reduced_kernel) * np.linalg.norm(explicit_kernel))
        correlation = float(abs(np.vdot(reduced_kernel, explicit_kernel)) / denominator)
        gain = np.vdot(reduced_kernel, explicit_kernel) / max(
            float(np.vdot(reduced_kernel, reduced_kernel).real), 1e-20
        )
        gain_corrected_error = float(
            np.linalg.norm(gain * reduced_kernel - explicit_kernel)
            / max(np.linalg.norm(explicit_kernel), 1e-20)
        )
        rows.append(
            {
                "frequency": frequency,
                "angle_deg": angle,
                "kernel_correlation": correlation,
                "gain_corrected_relative_error": gain_corrected_error,
            }
        )
    return {
        "status": "passed" if min(row["kernel_correlation"] for row in rows) >= 0.98 else "failed",
        "minimum_kernel_correlation": min(row["kernel_correlation"] for row in rows),
        "cases": rows,
    }


def _load_image(path: Path, size: int) -> np.ndarray:
    raw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(path)
    return cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA).astype(np.float64) / 255.0


def _synthetic_reference(size: int) -> np.ndarray:
    axis = np.arange(size)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    return normalize_image(
        ((x // max(size // 8, 1)) % 2).astype(float)
        + 0.6 * ((y // max(size // 5, 1)) % 2).astype(float)
    )


def _phase_transfer_stack(model: ReducedOrderModel, z_values_mm: list[float]) -> np.ndarray:
    n = model.size
    fx = np.fft.fftfreq(n, d=model.cfg.pixel_pitch_m)
    fy = np.fft.fftfreq(n, d=model.cfg.pixel_pitch_m)
    kx, ky = np.meshgrid(2.0 * np.pi * fx, 2.0 * np.pi * fy, indexing="xy")
    k = 2.0 * np.pi / model.cfg.wavelength_m
    kz = np.sqrt(np.maximum(k * k - kx * kx - ky * ky, 0.0))
    return np.stack([np.exp(1j * kz * value * 1e-3) for value in z_values_mm])


def _propagate_many(fields: np.ndarray, transfers: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft2(fields, axes=(-2, -1), norm="ortho")
    return np.fft.ifft2(
        spectrum[None, :, :, :] * transfers[:, None, :, :],
        axes=(-2, -1),
        norm="ortho",
    )


def _case_key(case: RobustnessCase) -> str:
    level = "inf" if math.isinf(case.level) else f"{case.level:g}"
    seed = "none" if case.seed is None else str(case.seed)
    return f"{case.family}:{level}:{seed}"


def _random_angle_tables(cases: list[RobustnessCase], count: int) -> np.ndarray:
    output = np.zeros((len(cases), count), dtype=np.float64)
    for index, case in enumerate(cases):
        rng = np.random.default_rng(int(case.seed))
        output[index] = rng.uniform(-case.level, case.level, count)
    return output


def _baseline_and_shared_errors(
    model: ReducedOrderModel,
    image: np.ndarray,
    points: list[tuple[int, int]],
    *,
    fmax: float,
    weights: np.ndarray,
    cases: dict[str, list[RobustnessCase]],
    progress_every: int,
) -> dict:
    n_points = len(points)
    white = float(image.sum())
    sqrt_image = np.sqrt(np.clip(image, 0.0, None))
    nominal = np.zeros((n_points, 4), dtype=np.float64)
    channels = np.zeros((n_points, 4, 5), dtype=np.float64)
    response_dc = np.zeros(n_points, dtype=np.complex128)
    response_gain = np.zeros(n_points, dtype=np.complex128)
    response_conjugate = np.zeros(n_points, dtype=np.complex128)

    angle_active = [row for row in cases["mask_angle"] if not row.is_baseline]
    frequency_active = [row for row in cases["mask_frequency"] if not row.is_baseline]
    defocus_active = [row for row in cases["conjugate_defocus"] if not row.is_baseline]
    angle_buckets = np.zeros((len(angle_active), n_points, 4), dtype=np.float64)
    frequency_buckets = np.zeros((len(frequency_active), n_points, 4), dtype=np.float64)
    defocus_buckets = np.zeros((len(defocus_active), n_points, 4), dtype=np.float64)

    high_count = sum(qx * qx + qy * qy > fmax * fmax for qx, qy in points)
    angle_draws = _random_angle_tables(angle_active, high_count)
    defocus_transfers = _phase_transfer_stack(
        model, [case.level for case in defocus_active]
    )
    high_index = 0
    for point_index, (qx, qy) in enumerate(points):
        target = np.asarray((qx, qy), dtype=np.float64)
        is_high = qx * qx + qy * qy > fmax * fmax
        if not is_high:
            quartet = model.quartet(target)
            bucket = np.sum(quartet * image[None, :, :], axis=(1, 2))
            nominal[point_index] = bucket
            channels[point_index, :, 1] = bucket
            kernel = four_step_kernel(quartet)
            angle_buckets[:, point_index] = bucket
            frequency_buckets[:, point_index] = bucket
            defocus_buckets[:, point_index] = bucket
        else:
            decomposition = parallel_frequency_decomposition(target, fmax)
            slm, slm_fields = model.quartet(
                decomposition["slm"], return_fields=True
            )
            difference = model.quartet(decomposition["difference"])
            mask_nominal = _mask(model, decomposition["mask"])
            channel_patterns = np.stack(
                (
                    slm * mask_nominal[None, :, :],
                    slm,
                    np.broadcast_to(mask_nominal, slm.shape),
                    difference,
                    np.ones_like(slm),
                ),
                axis=-1,
            )
            channel_bucket = np.sum(
                channel_patterns * image[None, :, :, None], axis=(1, 2)
            )
            channels[point_index] = channel_bucket
            effective = np.tensordot(channel_patterns, weights, axes=([-1], [0]))
            nominal[point_index] = channel_bucket @ weights
            kernel = four_step_kernel(effective)

            mask_vector = decomposition["mask"]
            mask_frequency = float(np.linalg.norm(mask_vector))
            base_angle = math.atan2(mask_vector[1], mask_vector[0])
            if angle_active:
                actual_angles = base_angle + np.deg2rad(angle_draws[:, high_index])
                masks = 0.5 + 0.5 * np.cos(
                    2.0
                    * np.pi
                    * mask_frequency
                    * (
                        np.cos(actual_angles)[:, None, None] * model.x
                        + np.sin(actual_angles)[:, None, None] * model.y
                    )
                )
                moire = np.einsum("pyx,cyx,yx->cp", slm, masks, image, optimize=True)
                mask_bucket = np.einsum("cyx,yx->c", masks, image, optimize=True)
                angle_buckets[:, point_index] = (
                    weights[0] * moire
                    + weights[1] * channel_bucket[:, 1][None, :]
                    + weights[2] * mask_bucket[:, None]
                    + weights[3] * channel_bucket[:, 3][None, :]
                    + weights[4] * white
                )
            if frequency_active:
                scales = 1.0 + np.asarray([case.level for case in frequency_active]) / 100.0
                vectors = scales[:, None] * mask_vector[None, :]
                masks = 0.5 + 0.5 * np.cos(
                    2.0
                    * np.pi
                    * (
                        vectors[:, 0, None, None] * model.x
                        + vectors[:, 1, None, None] * model.y
                    )
                )
                moire = np.einsum("pyx,cyx,yx->cp", slm, masks, image, optimize=True)
                mask_bucket = np.einsum("cyx,yx->c", masks, image, optimize=True)
                frequency_buckets[:, point_index] = (
                    weights[0] * moire
                    + weights[1] * channel_bucket[:, 1][None, :]
                    + weights[2] * mask_bucket[:, None]
                    + weights[3] * channel_bucket[:, 3][None, :]
                    + weights[4] * white
                )
            if defocus_active:
                propagated = _propagate_many(
                    slm_fields * sqrt_image[None, :, :], defocus_transfers
                )
                moire = np.sum(
                    np.abs(propagated) ** 2 * mask_nominal[None, None, :, :],
                    axis=(2, 3),
                )
                defocus_buckets[:, point_index] = (
                    weights[0] * moire
                    + weights[1] * channel_bucket[:, 1][None, :]
                    + weights[2] * channel_bucket[:, 2][None, :]
                    + weights[3] * channel_bucket[:, 3][None, :]
                    + weights[4] * white
                )
            high_index += 1

        response = response_from_kernel(kernel, qx, qy)
        response_dc[point_index] = response.dc
        response_gain[point_index] = response.gain
        response_conjugate[point_index] = response.conjugate_gain
        if progress_every and (
            point_index == 0
            or point_index + 1 == n_points
            or (point_index + 1) % progress_every == 0
        ):
            print(f"  nominal/shared errors: {point_index + 1}/{n_points}", flush=True)
    return {
        "nominal": nominal,
        "channels": channels,
        "response_dc": response_dc,
        "response_gain": response_gain,
        "response_conjugate": response_conjugate,
        "mask_angle": angle_buckets,
        "mask_frequency": frequency_buckets,
        "conjugate_defocus": defocus_buckets,
    }


def _nominal_kernels(
    model: ReducedOrderModel,
    points: list[tuple[int, int]],
    *,
    fmax: float,
    weights: np.ndarray,
    progress_every: int,
):
    for index, (qx, qy) in enumerate(points):
        target = np.asarray((qx, qy), dtype=np.float64)
        if qx * qx + qy * qy <= fmax * fmax:
            effective = model.quartet(target)
        else:
            decomposition = parallel_frequency_decomposition(target, fmax)
            slm = model.quartet(decomposition["slm"])
            difference = model.quartet(decomposition["difference"])
            mask = _mask(model, decomposition["mask"])
            features = np.stack(
                (
                    slm * mask[None, :, :],
                    slm,
                    np.broadcast_to(mask, slm.shape),
                    difference,
                    np.ones_like(slm),
                ),
                axis=-1,
            )
            effective = np.tensordot(features, weights, axes=([-1], [0]))
        if progress_every and (
            index == 0 or index + 1 == len(points) or (index + 1) % progress_every == 0
        ):
            print(f"  response matrix kernels: {index + 1}/{len(points)}", flush=True)
        yield four_step_kernel(effective)


def _static_phase_buckets(
    model: ReducedOrderModel,
    image: np.ndarray,
    points: list[tuple[int, int]],
    *,
    fmax: float,
    weights: np.ndarray,
    active_cases: list[RobustnessCase],
    progress_every: int,
) -> tuple[np.ndarray, list[dict]]:
    maps = np.stack(
        [
            zero_mean_rms_map(model.size, case.level * np.pi, int(case.seed))
            for case in active_cases
        ]
    )
    buckets = np.zeros((len(active_cases), len(points), 4), dtype=np.float64)
    image_case = image[None, None, :, :]
    for point_index, (qx, qy) in enumerate(points):
        target = np.asarray((qx, qy), dtype=np.float64)
        is_high = qx * qx + qy * qy > fmax * fmax
        if not is_high:
            quartet = model.quartet(target, phase_maps=maps)
            buckets[:, point_index] = np.sum(quartet * image_case, axis=(2, 3))
        else:
            decomposition = parallel_frequency_decomposition(target, fmax)
            slm = model.quartet(decomposition["slm"], phase_maps=maps)
            difference = model.quartet(decomposition["difference"], phase_maps=maps)
            mask = _mask(model, decomposition["mask"])
            moire = np.sum(slm * mask[None, None, :, :] * image_case, axis=(2, 3))
            slm_bucket = np.sum(slm * image_case, axis=(2, 3))
            difference_bucket = np.sum(difference * image_case, axis=(2, 3))
            mask_bucket = float(np.sum(mask * image))
            buckets[:, point_index] = (
                weights[0] * moire
                + weights[1] * slm_bucket
                + weights[2] * mask_bucket
                + weights[3] * difference_bucket
                + weights[4] * float(image.sum())
            )
        if progress_every and (
            point_index == 0
            or point_index + 1 == len(points)
            or (point_index + 1) % progress_every == 0
        ):
            print(f"  static phase: {point_index + 1}/{len(points)}", flush=True)
    realizations = [
        {
            "case": _case_key(case),
            "rms_rad": float(case.level * np.pi),
            "min_rad": float(values.min()),
            "max_rad": float(values.max()),
            "sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
        }
        for case, values in zip(active_cases, maps)
    ]
    return buckets, realizations


def _derive_detector_noise(
    nominal: np.ndarray, active_cases: list[RobustnessCase]
) -> tuple[np.ndarray, list[dict]]:
    ac_scale = float(np.std(nominal - nominal.mean(axis=1, keepdims=True)))
    matrices = []
    details = []
    for case in active_cases:
        sigma = ac_scale / (10.0 ** (case.level / 20.0))
        rng = np.random.default_rng(int(case.seed))
        matrices.append(nominal + rng.normal(0.0, sigma, nominal.shape))
        details.append({"case": _case_key(case), "sigma": sigma, "ac_scale": ac_scale})
    return np.stack(matrices), details


def _derive_calibration_mismatch(
    nominal: np.ndarray,
    channels: np.ndarray,
    points: list[tuple[int, int]],
    *,
    fmax: float,
    weights: np.ndarray,
    active_cases: list[RobustnessCase],
) -> tuple[np.ndarray, list[dict]]:
    output = np.zeros((len(active_cases), *nominal.shape), dtype=np.float64)
    details = []
    high = np.asarray([qx * qx + qy * qy > fmax * fmax for qx, qy in points])
    for index, case in enumerate(active_cases):
        drift = calibration_drift(case.level, int(case.seed))
        gains = 1.0 + drift
        output[index] = nominal
        output[index, high] = np.einsum(
            "npc,c->np", channels[high] * gains[None, None, :], weights, optimize=True
        )
        output[index, ~high] = nominal[~high] * gains[1]
        details.append(
            {"case": _case_key(case), "channel_gain": gains.tolist(), "rms_percent": case.level}
        )
    return output, details


def _reconstruct_bucket_cases(
    buckets: np.ndarray,
    points: list[tuple[int, int]],
    shared: dict,
    *,
    image: np.ndarray,
    ideal: np.ndarray,
    response_solver: CalibratedResponseSolver | None = None,
) -> list[dict]:
    values = np.asarray(buckets, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, :, :]
    raw = four_step_coefficients(values)
    dc_response = shared["response_dc"][None, :]
    gain = shared["response_gain"][None, :]
    conjugate = shared["response_conjugate"][None, :]
    white = float(image.sum())
    corrected = raw - dc_response * white
    denominator = np.abs(gain) ** 2 - np.abs(conjugate) ** 2
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError("nominal response contains a singular frequency")
    coefficients = (np.conj(gain) * corrected - conjugate * np.conj(corrected)) / denominator
    if response_solver is not None:
        coefficients = response_solver.solve(coefficients)
    rows = []
    for coefficient in coefficients:
        reconstruction, spectrum = coefficients_to_reconstruction(
            image.shape[0], points, coefficient, dc=white
        )
        rows.append(
            {
                "reconstruction": reconstruction,
                "spectrum": spectrum,
                "metrics_full": image_metrics(image, reconstruction),
                "metrics_matched": image_metrics(ideal, reconstruction),
            }
        )
    return rows


def _serialize_config(cfg: PhysicalBandwidthConfig) -> dict:
    values = asdict(cfg)
    values["thresholds"] = asdict(cfg.thresholds)
    return values


def run_robustness_pipeline(
    output_dir: Path,
    *,
    quick: bool = False,
    resume: bool = True,
    image_path: Path | None = None,
    response_matrix_source: Path | None = None,
    progress_every: int | None = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if quick:
        cfg = replace(
            PhysicalBandwidthConfig.quick(),
            slm_pixels=16,
            oversample=2,
            fft_size=64,
            fill_factor=0.95,
        )
        fmax, carrier, iris_mm = 2.0, 4.0, 2.0
        image = _synthetic_reference(cfg.slm_pixels)
        downsample = 16
        progress = 0 if progress_every is None else progress_every
    else:
        cfg = PhysicalBandwidthConfig.production_256()
        fmax, carrier, iris_mm = 32.0, 100.0, 1.75
        if image_path is None:
            image_path = Path(__file__).resolve().parent.parent / "Test_image" / "USAF-1951.jpg"
        image = _load_image(Path(image_path).resolve(), cfg.slm_pixels)
        downsample = 64
        progress = 250 if progress_every is None else progress_every
    model = ReducedOrderModel(cfg, carrier=carrier, iris_mm=iris_mm)
    cases = build_robustness_cases()
    points = half_plane_frequency_points(2.0 * fmax)
    ideal = _bandlimited_reference(image, 2.0 * fmax)
    ideal_spi = _bandlimited_reference(image, fmax)

    weights_path = output / "global_weights.json"
    if resume and weights_path.exists():
        weight_info = json.loads(weights_path.read_text(encoding="utf-8"))
        weights = np.asarray(weight_info["direct_weights"], dtype=np.float64)
        weight_diagnostics = weight_info["diagnostics"]
    else:
        print("Fitting one object-independent global five-channel vector...", flush=True)
        weights, weight_diagnostics = fit_reduced_global_weights(
            model, fmax=fmax, downsample=downsample, iterations=2 if quick else 4
        )
        weights_path.write_text(
            json.dumps(
                {"direct_weights": weights.tolist(), "diagnostics": weight_diagnostics},
                indent=2,
            ),
            encoding="utf-8",
        )
    calibration_validation = evaluate_reduced_global_weights(
        model, weights, fmax=fmax, downsample=downsample
    )

    crosscheck_path = output / "model_crosscheck.json"
    if quick:
        model_crosscheck = {"status": "skipped_in_quick_mode", "cases": []}
    elif resume and crosscheck_path.exists():
        model_crosscheck = json.loads(crosscheck_path.read_text(encoding="utf-8"))
    else:
        print("Cross-checking the reduced model against explicit carrier/order selection...", flush=True)
        model_crosscheck = crosscheck_reduced_model(
            cfg, model, carrier=carrier, iris_mm=iris_mm
        )
        crosscheck_path.write_text(
            json.dumps(model_crosscheck, indent=2), encoding="utf-8"
        )

    shared_path = output / "shared_measurements.npz"
    if resume and shared_path.exists():
        print("Reusing nominal, mask-error, and defocus measurements.", flush=True)
        with np.load(shared_path) as cache:
            shared = {name: cache[name] for name in cache.files}
    else:
        print("Acquiring nominal, mask-error, and wave-optics defocus measurements...", flush=True)
        shared = _baseline_and_shared_errors(
            model,
            image,
            points,
            fmax=fmax,
            weights=weights,
            cases=cases,
            progress_every=progress,
        )
        np.savez_compressed(shared_path, **shared)

    response_matrix_path = (
        Path(response_matrix_source).resolve()
        if response_matrix_source is not None
        else output / "nominal_response_matrix.npz"
    )
    if response_matrix_path.exists() and (resume or response_matrix_source is not None):
        print("Reusing nominal sparse system-response matrix.", flush=True)
        response_matrix = sparse.load_npz(response_matrix_path)
    else:
        print("Building nominal sparse system-response matrix...", flush=True)
        responses = [
            FrozenResponse(dc, gain, conjugate)
            for dc, gain, conjugate in zip(
                shared["response_dc"], shared["response_gain"], shared["response_conjugate"]
            )
        ]
        response_matrix = matrix_from_kernels(
            _nominal_kernels(
                model,
                points,
                fmax=fmax,
                weights=weights,
                progress_every=progress,
            ),
            points,
            responses,
            max_terms=16 if quick else 96,
            relative_threshold=1e-3 if quick else 5e-4,
        )
        response_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(response_matrix_path, response_matrix)
    print(
        f"Factoring sparse response matrix ({response_matrix.shape[0]} x {response_matrix.shape[1]}, "
        f"{response_matrix.nnz} nonzeros)...",
        flush=True,
    )
    response_solver = CalibratedResponseSolver(response_matrix)

    phase_active = [row for row in cases["slm_static_phase"] if not row.is_baseline]
    phase_path = output / "slm_static_phase_measurements.npz"
    if resume and phase_path.exists():
        print("Reusing static-phase measurements.", flush=True)
        with np.load(phase_path, allow_pickle=False) as cache:
            phase_buckets = cache["buckets"]
        phase_realizations = json.loads(
            (output / "slm_static_phase_realizations.json").read_text(encoding="utf-8")
        )
    else:
        print("Acquiring static SLM phase-error measurements...", flush=True)
        phase_buckets, phase_realizations = _static_phase_buckets(
            model,
            image,
            points,
            fmax=fmax,
            weights=weights,
            active_cases=phase_active,
            progress_every=progress,
        )
        np.savez_compressed(phase_path, buckets=phase_buckets)
        (output / "slm_static_phase_realizations.json").write_text(
            json.dumps(phase_realizations, indent=2), encoding="utf-8"
        )

    active_by_family = {
        family: [row for row in rows if not row.is_baseline]
        for family, rows in cases.items()
    }
    detector_buckets, detector_details = _derive_detector_noise(
        shared["nominal"], active_by_family["detector_noise"]
    )
    calibration_buckets, calibration_details = _derive_calibration_mismatch(
        shared["nominal"],
        shared["channels"],
        points,
        fmax=fmax,
        weights=weights,
        active_cases=active_by_family["calibration_mismatch"],
    )
    bucket_matrices = {
        "mask_angle": shared["mask_angle"],
        "mask_frequency": shared["mask_frequency"],
        "slm_static_phase": phase_buckets,
        "detector_noise": detector_buckets,
        "calibration_mismatch": calibration_buckets,
        "conjugate_defocus": shared["conjugate_defocus"],
    }
    baseline_result = _reconstruct_bucket_cases(
        shared["nominal"],
        points,
        shared,
        image=image,
        ideal=ideal,
        response_solver=response_solver,
    )[0]
    low_indices = np.asarray(
        [index for index, (qx, qy) in enumerate(points) if qx * qx + qy * qy <= fmax * fmax],
        dtype=np.int64,
    )
    low_points = [points[index] for index in low_indices]
    low_shared = {
        name: shared[name][low_indices]
        for name in ("response_dc", "response_gain", "response_conjugate")
    }
    spi_baseline = _reconstruct_bucket_cases(
        shared["nominal"][low_indices],
        low_points,
        low_shared,
        image=image,
        ideal=ideal_spi,
        response_solver=None,
    )[0]
    family_results: dict[str, list[dict]] = {}
    for family, family_cases in cases.items():
        active_results = _reconstruct_bucket_cases(
            bucket_matrices[family],
            points,
            shared,
            image=image,
            ideal=ideal,
            response_solver=response_solver,
        )
        rows = []
        active_index = 0
        for case in family_cases:
            reconstruction = baseline_result if case.is_baseline else active_results[active_index]
            if not case.is_baseline:
                active_index += 1
            rows.append(
                {
                    "case": case,
                    **reconstruction,
                }
            )
        family_results[family] = rows

    configuration = {
        "fmax": fmax,
        "msf_radius": 2.0 * fmax,
        "carrier": carrier,
        "iris_mm": iris_mm,
        "reduced_pupil_cutoff_cycles": model.cutoff_cycles,
        "physical": _serialize_config(cfg),
        "model": "validated reduced-order finite-pupil labelled +1-order wave optics",
    }
    result = {
        "configuration": configuration,
        "reference": image,
        "ideal_spi_reference": ideal_spi,
        "ideal_msf_reference": ideal,
        "global_weights": weights,
        "weight_diagnostics": weight_diagnostics,
        "calibration_validation": calibration_validation,
        "model_crosscheck": model_crosscheck,
        "response_matrix": {
            "shape": list(response_matrix.shape),
            "nonzeros": int(response_matrix.nnz),
            "max_terms_per_complex_row": 16 if quick else 96,
            "relative_threshold": 1e-3 if quick else 5e-4,
        },
        "points": points,
        "baseline": baseline_result,
        "spi_baseline": spi_baseline,
        "family_results": family_results,
        "separate_slm_and_difference_measurements": True,
        "realizations": {
            "slm_static_phase": phase_realizations,
            "detector_noise": detector_details,
            "calibration_mismatch": calibration_details,
        },
    }
    np.savez_compressed(
        output / "pipeline_cache.npz",
        reference=image,
        ideal_spi_reference=ideal_spi,
        ideal_msf_reference=ideal,
        spi_baseline_reconstruction=spi_baseline["reconstruction"],
        spi_baseline_spectrum=spi_baseline["spectrum"],
        baseline_reconstruction=baseline_result["reconstruction"],
        baseline_spectrum=baseline_result["spectrum"],
        global_weights=weights,
    )
    return result
