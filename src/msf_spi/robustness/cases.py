from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from ..physical.propagation import gaussian_amplitude
from ..metrics import normalize_image
from .errors import (
    ANGLE_HALF_WIDTH_DEG,
    CALIBRATION_RMS_PERCENT,
    DEFOCUS_MM,
    DETECTOR_AC_SNR_DB,
    MASK_FREQUENCY_PERCENT,
    RANDOM_SEEDS,
    SLM_PHASE_RMS_PI,
)
from ..physical.config import PhysicalSLMConfig as PhysicalBandwidthConfig
from ..reconstruction.fourier import centered_to_fft_coefficient


PHASES_DEG = np.asarray((0.0, 90.0, 180.0, 270.0), dtype=np.float64)


@dataclass(frozen=True)
class RobustnessCase:
    family: str
    level: float
    unit: str
    seed: int | None
    label: str

    @property
    def is_baseline(self) -> bool:
        if self.family == "detector_noise":
            return math.isinf(self.level)
        return self.level == 0.0


@dataclass(frozen=True)
class FrozenResponse:
    dc: complex
    gain: complex
    conjugate_gain: complex


def _stochastic_cases(
    family: str,
    levels: Iterable[float],
    unit: str,
    label,
) -> list[RobustnessCase]:
    values = [float(value) for value in levels]
    rows = [RobustnessCase(family, values[0], unit, None, label(values[0]))]
    for level in values[1:]:
        for seed in RANDOM_SEEDS:
            rows.append(RobustnessCase(family, level, unit, int(seed), label(level)))
    return rows


def build_robustness_cases() -> dict[str, list[RobustnessCase]]:
    return {
        "mask_angle": _stochastic_cases(
            "mask_angle",
            ANGLE_HALF_WIDTH_DEG,
            "uniform half-width (deg)",
            lambda value: f"uniform +/-{value:g} deg",
        ),
        "mask_frequency": [
            RobustnessCase(
                "mask_frequency", float(value), "frequency error (%)", None, f"{value:+g}%"
            )
            for value in MASK_FREQUENCY_PERCENT
        ],
        "slm_static_phase": _stochastic_cases(
            "slm_static_phase",
            SLM_PHASE_RMS_PI,
            "phase RMS (pi rad)",
            lambda value: f"RMS={value:g} pi rad",
        ),
        "detector_noise": _stochastic_cases(
            "detector_noise",
            DETECTOR_AC_SNR_DB,
            "equivalent bucket AC-SNR (dB)",
            lambda value: "noiseless" if math.isinf(value) else f"{value:g} dB",
        ),
        "calibration_mismatch": _stochastic_cases(
            "calibration_mismatch",
            CALIBRATION_RMS_PERCENT,
            "relative channel-gain RMS (%)",
            lambda value: f"RMS={value:g}%",
        ),
        "conjugate_defocus": [
            RobustnessCase(
                "conjugate_defocus", float(value), "axial offset (mm)", None, f"z={value:g} mm"
            )
            for value in DEFOCUS_MM
        ],
    }


def zero_mean_rms_map(size: int, rms: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((size, size))
    values -= values.mean()
    scale = float(np.sqrt(np.mean(values**2)))
    if rms == 0.0 or scale <= 0.0:
        return np.zeros((size, size), dtype=np.float64)
    return values * (float(rms) / scale)


class ReducedOrderModel:
    """Finite-pupil labelled +1-order model on the physical SLM pixel grid.

    The explicit padded carrier calculation is used separately for validation.
    This operator retains the measured iris bandwidth, relay NA, Gaussian input,
    pixel pitch, and finite fill-factor transfer while avoiding irrelevant empty
    padding during thousands of reconstruction measurements.
    """

    def __init__(
        self,
        cfg: PhysicalBandwidthConfig,
        *,
        carrier: float,
        iris_mm: float,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.carrier = float(carrier)
        self.iris_mm = float(iris_mm)
        self.size = int(cfg.slm_pixels)
        axis = np.arange(self.size, dtype=np.float64) + 0.5 - self.size / 2.0
        self.x, self.y = np.meshgrid(axis / self.size, axis / self.size, indexing="xy")
        frequency_axis = np.fft.fftshift(np.fft.fftfreq(self.size) * self.size)
        self.qx, self.qy = np.meshgrid(frequency_axis, frequency_axis, indexing="xy")
        iris_cycles = self.iris_mm * 1e-3 / cfg.fourier_plane_m_per_cycle
        relay_cycles = min(cfg.order_lens_na, cfg.relay_na) * cfg.active_width_m / cfg.wavelength_m
        self.cutoff_cycles = float(min(iris_cycles, relay_cycles))
        pupil = self.qx**2 + self.qy**2 <= self.cutoff_cycles**2 + 1e-12
        active_side = cfg.pixel_pitch_m * math.sqrt(cfg.fill_factor)
        abs_fx = (self.carrier + self.qx) / cfg.active_width_m
        abs_fy = self.qy / cfg.active_width_m
        pixel_transfer = np.sinc(active_side * abs_fx) * np.sinc(active_side * abs_fy)
        reference_transfer = float(np.sinc(active_side * self.carrier / cfg.active_width_m))
        if abs(reference_transfer) < 1e-9:
            raise ValueError("carrier lies at a finite-pixel aperture zero")
        self.transfer = pupil * (pixel_transfer / reference_transfer)
        self.incident = gaussian_amplitude(
            self.size, cfg.pixel_pitch_m, cfg.gaussian_radius_m
        )
        self.incident_floor = float(self.incident.min())

    def target_intensities(self, vector: np.ndarray) -> np.ndarray:
        q = np.asarray(vector, dtype=np.float64)
        if q.shape != (2,):
            raise ValueError("frequency vector must contain qx and qy")
        argument = 2.0 * np.pi * (q[0] * self.x + q[1] * self.y)
        return 0.5 + 0.5 * np.cos(
            argument[None, :, :] + np.deg2rad(PHASES_DEG)[:, None, None]
        )

    def _desired_fields(self, vector: np.ndarray) -> np.ndarray:
        target = self.target_intensities(vector)
        command = np.sqrt(np.clip(target, 0.0, 1.0)) / np.maximum(self.incident, 1e-12)
        command = np.clip(command * self.incident_floor, 0.0, 1.0)
        return self.incident[None, :, :] * command

    def quartet(
        self,
        vector: np.ndarray,
        *,
        phase_maps: np.ndarray | None = None,
        return_fields: bool = False,
    ):
        desired = self._desired_fields(vector)
        if phase_maps is None:
            fields = desired.astype(np.complex128)
        else:
            maps = np.asarray(phase_maps, dtype=np.float64)
            if maps.ndim == 2:
                maps = maps[None, :, :]
            if maps.ndim != 3 or maps.shape[1:] != (self.size, self.size):
                raise ValueError("phase maps must have shape (case, y, x)")
            fields = desired[None, :, :, :] * np.exp(1j * maps[:, None, :, :])
        spectrum = np.fft.fftshift(
            np.fft.fft2(fields, axes=(-2, -1), norm="ortho"), axes=(-2, -1)
        )
        recovered = np.fft.ifft2(
            np.fft.ifftshift(spectrum * self.transfer, axes=(-2, -1)),
            axes=(-2, -1),
            norm="ortho",
        )
        intensity = np.abs(recovered) ** 2
        phase_mean = intensity.mean(axis=-3)
        floor = np.maximum(2.0 * phase_mean, max(float(phase_mean.max()) * 1e-12, 1e-20))
        normalized_intensity = intensity / floor[..., None, :, :]
        normalized_field = recovered / np.sqrt(floor)[..., None, :, :]
        if phase_maps is None:
            normalized_intensity = normalized_intensity.astype(np.float32)
            normalized_field = normalized_field.astype(np.complex64)
        else:
            normalized_intensity = normalized_intensity.astype(np.float32)
            normalized_field = normalized_field.astype(np.complex64)
        if return_fields:
            return normalized_intensity, normalized_field
        return normalized_intensity


def response_from_kernel(kernel: np.ndarray, qx: int, qy: int) -> FrozenResponse:
    values = np.asarray(kernel, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("kernel must be a square two-dimensional array")
    size = values.shape[0]
    axis = np.arange(size, dtype=np.float64) + 0.5 - size / 2.0
    x, y = np.meshgrid(axis, axis, indexing="xy")
    target = np.exp(-2j * np.pi * (qx * x + qy * y) / size)
    dc = complex(values.mean())
    zero_mean = values - dc
    sample_count = float(values.size)
    gain = complex(np.vdot(target, zero_mean) / sample_count)
    conjugate_gain = complex(np.vdot(np.conj(target), zero_mean) / sample_count)
    return FrozenResponse(dc=dc, gain=gain, conjugate_gain=conjugate_gain)


def apply_frozen_response(
    raw_coefficient: complex,
    response: FrozenResponse,
    white_bucket: float,
) -> complex:
    denominator = abs(response.gain) ** 2 - abs(response.conjugate_gain) ** 2
    if abs(denominator) < 1e-12:
        raise ValueError("frozen system response is singular")
    corrected = complex(raw_coefficient) - response.dc * float(white_bucket)
    return (
        np.conj(response.gain) * corrected
        - response.conjugate_gain * np.conj(corrected)
    ) / denominator


def coefficients_to_reconstruction(
    size: int,
    points: Iterable[tuple[int, int]],
    coefficients: np.ndarray,
    *,
    dc: float,
) -> tuple[np.ndarray, np.ndarray]:
    point_list = list(points)
    values = np.asarray(coefficients, dtype=np.complex128)
    if len(point_list) != values.size:
        raise ValueError("point and coefficient counts do not match")
    spectrum = np.zeros((size, size), dtype=np.complex128)
    spectrum[0, 0] = float(dc)
    for (qx, qy), value in zip(point_list, values):
        converted = centered_to_fft_coefficient(value, qx, qy, size)
        spectrum[qy % size, qx % size] = converted
        spectrum[-qy % size, -qx % size] = np.conj(converted)
    reconstruction = normalize_image(np.real(np.fft.ifft2(spectrum)))
    return reconstruction, spectrum


def four_step_kernel(patterns: np.ndarray) -> np.ndarray:
    values = np.asarray(patterns)
    if values.shape[0] != 4:
        raise ValueError("four phase patterns are required")
    return (values[0] - values[2]) + 1j * (values[1] - values[3])


def four_step_coefficients(buckets: np.ndarray) -> np.ndarray:
    values = np.asarray(buckets, dtype=np.float64)
    if values.shape[-1] != 4:
        raise ValueError("last bucket dimension must contain four phases")
    return (values[..., 0] - values[..., 2]) + 1j * (
        values[..., 1] - values[..., 3]
    )


def angular_spectrum_propagate(
    fields: np.ndarray,
    z_m: float,
    *,
    wavelength_m: float,
    dx_m: float,
) -> np.ndarray:
    values = np.asarray(fields, dtype=np.complex128)
    if z_m == 0.0:
        return values.copy()
    height, width = values.shape[-2:]
    fy = np.fft.fftfreq(height, d=dx_m)
    fx = np.fft.fftfreq(width, d=dx_m)
    kx, ky = np.meshgrid(2.0 * np.pi * fx, 2.0 * np.pi * fy, indexing="xy")
    k = 2.0 * np.pi / wavelength_m
    kz = np.sqrt(np.maximum(k * k - kx * kx - ky * ky, 0.0))
    transfer = np.exp(1j * kz * float(z_m))
    return np.fft.ifft2(
        np.fft.fft2(values, axes=(-2, -1), norm="ortho") * transfer,
        axes=(-2, -1),
        norm="ortho",
    )
