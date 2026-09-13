from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from .config import PhysicalSLMConfig as PhysicalBandwidthConfig, QualityThresholds
from .propagation import _inverse_sinc_nonnegative, gaussian_amplitude


@dataclass(frozen=True)
class PatternQuality:
    correlation: float
    leakage: float
    modulation_error: float
    fitted_phase_error_deg: float
    passes: bool


@dataclass(frozen=True)
class PhysicalDiagnostics:
    order_center_mm: float
    input_power: float
    selected_power: float
    relayed_power: float
    selected_efficiency: float
    relay_cutoff_cycles: float


@dataclass
class RecoveredPattern:
    target_intensity: np.ndarray
    slm_phase: np.ndarray
    full_spectrum: np.ndarray
    iris: np.ndarray
    selected_spectrum: np.ndarray
    recovered_field: np.ndarray
    recovered_intensity: np.ndarray
    quality: PatternQuality
    diagnostics: PhysicalDiagnostics


def normalize_phase_sequence(intensities: np.ndarray) -> np.ndarray:
    """Remove the phase-independent illumination envelope from a phase quartet.

    The four sinusoidal phase states have a local mean equal to one half of the
    all-pass illumination.  Dividing by twice that mean preserves the physical
    fringe contrast while removing only the common envelope used by four-step
    demodulation.
    """
    values = np.asarray(intensities, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("intensities must have shape (phase, y, x)")
    phase_mean = values.mean(axis=0)
    floor = max(float(phase_mean.max()) * 1e-12, 1e-20)
    return values / (2.0 * np.maximum(phase_mean, floor))[None, :, :]


def target_intensity(
    size: int,
    frequency: float,
    angle_deg: float,
    phase_deg: float,
) -> np.ndarray:
    axis = np.arange(size, dtype=np.float64) + 0.5 - size / 2.0
    x, y = np.meshgrid(axis, axis, indexing="xy")
    theta = math.radians(angle_deg)
    argument = (
        2.0
        * np.pi
        * float(frequency)
        * (x * math.cos(theta) + y * math.sin(theta))
        / size
        + math.radians(phase_deg)
    )
    return 0.5 + 0.5 * np.cos(argument)


def phase_only_hologram_cycles(
    target_amplitude: np.ndarray,
    target_phase: np.ndarray,
    carrier_cycles: float,
) -> dict[str, np.ndarray | float]:
    amplitude = np.asarray(target_amplitude, dtype=np.float64)
    phase = np.asarray(target_phase, dtype=np.float64)
    if amplitude.ndim != 2 or phase.shape != amplitude.shape:
        raise ValueError("target amplitude and phase must be matching two-dimensional arrays")
    if np.any((amplitude < 0.0) | (amplitude > 1.0)):
        raise ValueError("normalized target amplitude must lie in [0, 1]")
    q = -_inverse_sinc_nonnegative(amplitude)
    modulation_depth = 1.0 + q / np.pi
    phase_function = phase - np.pi * modulation_depth
    x = np.arange(amplitude.shape[1], dtype=np.float64)
    carrier = 2.0 * np.pi * float(carrier_cycles) * x / amplitude.shape[1]
    wrapped = np.mod(phase_function + carrier[None, :], 2.0 * np.pi)
    slm_phase = modulation_depth * wrapped
    return {
        "slm_phase": slm_phase,
        "slm_field": np.exp(1j * slm_phase),
        "carrier_cycles": float(carrier_cycles),
    }


def hold_pixels(values: np.ndarray, oversample: int) -> np.ndarray:
    if oversample < 1:
        raise ValueError("oversample must be positive")
    return np.repeat(np.repeat(np.asarray(values), oversample, axis=0), oversample, axis=1)


def _center_pad(values: np.ndarray, size: int) -> tuple[np.ndarray, tuple[slice, slice]]:
    height, width = values.shape
    if height > size or width > size:
        raise ValueError("padded size is smaller than the active array")
    y0 = (size - height) // 2
    x0 = (size - width) // 2
    region = (slice(y0, y0 + height), slice(x0, x0 + width))
    output = np.zeros((size, size), dtype=values.dtype)
    output[region] = values
    return output, region


@lru_cache(maxsize=8)
def _frequency_geometry(cfg: PhysicalBandwidthConfig):
    dx = cfg.pixel_pitch_m / cfg.oversample
    spatial = np.fft.fftshift(np.fft.fftfreq(cfg.fft_size, d=dx))
    fx, fy = np.meshgrid(spatial, spatial, indexing="xy")
    focal_x = cfg.wavelength_m * cfg.order_lens_focal_m * fx
    focal_y = cfg.wavelength_m * cfg.order_lens_focal_m * fy
    coordinate = (np.arange(cfg.fft_size, dtype=np.float64) - cfg.fft_size / 2.0) * dx
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    return fx, fy, focal_x, focal_y, x, y


def _full_pixel_aperture_transfer(
    cfg: PhysicalBandwidthConfig,
) -> np.ndarray:
    fx, fy, *_ = _frequency_geometry(cfg)
    return np.sinc(cfg.pixel_pitch_m * fx) * np.sinc(cfg.pixel_pitch_m * fy)


def pixel_aperture_transfer_ratio(cfg: PhysicalBandwidthConfig) -> np.ndarray:
    """Return the finite-fill selected-order transfer relative to pixel hold.

    The existing oversampled field uses a full-pixel zero-order hold.  A square
    active aperture with area fraction ``fill_factor`` has side length
    ``pixel_pitch * sqrt(fill_factor)``.  This helper supplies the continuous
    rectangular-aperture correction relative to the full-pixel hold.  Transfer
    zeros are set to zero here and rejected explicitly if a selected iris
    intersects them.
    """
    cfg.validate()
    if cfg.fill_factor == 1.0:
        return np.ones((cfg.fft_size, cfg.fft_size), dtype=np.float64)
    fx, fy, *_ = _frequency_geometry(cfg)
    active_width = cfg.pixel_pitch_m * math.sqrt(cfg.fill_factor)
    numerator = cfg.fill_factor * np.sinc(active_width * fx) * np.sinc(
        active_width * fy
    )
    denominator = _full_pixel_aperture_transfer(cfg)
    ratio = np.zeros_like(numerator)
    valid = np.abs(denominator) > 1e-12
    np.divide(numerator, denominator, out=ratio, where=valid)
    return ratio


def _phase_only_slm_field(
    target_low: np.ndarray,
    carrier_frequency: float,
    cfg: PhysicalBandwidthConfig,
) -> tuple[np.ndarray, np.ndarray]:
    incident_low = gaussian_amplitude(
        cfg.slm_pixels, cfg.pixel_pitch_m, cfg.gaussian_radius_m
    )
    command = np.sqrt(np.clip(target_low, 0.0, 1.0)) / np.maximum(incident_low, 1e-9)
    # Use one physical amplitude scale for every phase state.  Normalizing each
    # pattern by its own maximum destroys the relative intensity of low- or
    # zero-frequency phase steps (for example 1, 0.5, 0, 0.5 at f=0).
    command *= float(incident_low.min())
    command = np.clip(command, 0.0, 1.0)
    encoded = phase_only_hologram_cycles(
        command, np.zeros_like(command), carrier_frequency
    )
    incident = hold_pixels(incident_low, cfg.oversample)
    phase = hold_pixels(np.asarray(encoded["slm_phase"]), cfg.oversample)
    return incident * np.exp(1j * phase), phase


def _propagate_phase_field(
    active_field: np.ndarray,
    carrier_frequency: float,
    iris_radius_mm: float,
    cfg: PhysicalBandwidthConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PhysicalDiagnostics]:
    padded, active_region = _center_pad(active_field, cfg.fft_size)
    spectrum = np.fft.fftshift(np.fft.fft2(padded, norm="ortho"))
    fx, fy, focal_x, focal_y, x, _ = _frequency_geometry(cfg)
    order_center_m = carrier_frequency * cfg.fourier_plane_m_per_cycle
    if iris_radius_mm <= 0:
        iris = np.zeros_like(focal_x, dtype=bool)
    else:
        radius_m = iris_radius_mm * 1e-3
        iris = (focal_x - order_center_m) ** 2 + focal_y**2 <= radius_m**2
    full_pixel_transfer = _full_pixel_aperture_transfer(cfg)
    if np.any(iris & (np.abs(full_pixel_transfer) <= 1e-12)):
        raise ValueError("selected +1-order iris intersects a full-pixel aperture zero")
    selected_spectrum = spectrum * iris * pixel_aperture_transfer_ratio(cfg)
    tilted = np.fft.ifft2(np.fft.ifftshift(selected_spectrum), norm="ortho")
    recentered = tilted * np.exp(
        -2j * np.pi * carrier_frequency * x / cfg.active_width_m
    )
    recentered_spectrum = np.fft.fftshift(np.fft.fft2(recentered, norm="ortho"))
    cutoff = min(cfg.order_lens_na, cfg.relay_na) / cfg.wavelength_m
    relay_pupil = fx**2 + fy**2 <= cutoff**2
    relayed_spectrum = recentered_spectrum * relay_pupil
    relayed = np.fft.ifft2(np.fft.ifftshift(relayed_spectrum), norm="ortho")

    input_power = float(np.sum(np.abs(padded) ** 2))
    selected_power = float(np.sum(np.abs(selected_spectrum) ** 2))
    relayed_power = float(np.sum(np.abs(relayed) ** 2))
    diagnostics = PhysicalDiagnostics(
        order_center_mm=float(order_center_m * 1e3),
        input_power=input_power,
        selected_power=selected_power,
        relayed_power=relayed_power,
        selected_efficiency=selected_power / input_power if input_power > 0 else 0.0,
        relay_cutoff_cycles=float(cutoff * cfg.active_width_m),
    )
    return spectrum, iris, selected_spectrum, relayed[active_region], diagnostics


def _wrapped_angle_deg(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def sinusoid_quality(
    target: np.ndarray,
    recovered: np.ndarray,
    frequency: float,
    angle_deg: float,
    phase_deg: float,
    thresholds: QualityThresholds,
) -> PatternQuality:
    target_values = np.asarray(target, dtype=np.float64)
    recovered_values = np.asarray(recovered, dtype=np.float64)
    if target_values.shape != recovered_values.shape or target_values.ndim != 2:
        raise ValueError("target and recovered intensity must have matching 2-D shapes")
    height, width = target_values.shape
    x = (np.arange(width, dtype=np.float64) + 0.5 - width / 2.0) / width
    y = (np.arange(height, dtype=np.float64) + 0.5 - height / 2.0) / height
    xx, yy = np.meshgrid(x, y, indexing="xy")
    theta = math.radians(angle_deg)
    argument = 2.0 * np.pi * frequency * (
        xx * math.cos(theta) + yy * math.sin(theta)
    )
    design = np.column_stack(
        (np.ones(target_values.size), np.cos(argument).ravel(), np.sin(argument).ravel())
    )
    coeff_target = np.linalg.lstsq(design, target_values.ravel(), rcond=None)[0]
    coeff_recovered = np.linalg.lstsq(design, recovered_values.ravel(), rcond=None)[0]
    fitted = (design @ coeff_recovered).reshape(recovered_values.shape)

    t_ac = target_values - target_values.mean()
    r_ac = recovered_values - recovered_values.mean()
    denominator = float(np.linalg.norm(t_ac) * np.linalg.norm(r_ac))
    correlation = float(np.sum(t_ac * r_ac) / denominator) if denominator > 0 else -1.0
    total_ac_energy = float(np.sum(r_ac**2))
    residual = recovered_values - fitted
    leakage = (
        float(np.sum(residual**2)) / total_ac_energy
        if total_ac_energy > 1e-20
        else float("inf")
    )
    target_modulation = float(
        np.hypot(coeff_target[1], coeff_target[2]) / max(abs(coeff_target[0]), 1e-20)
    )
    recovered_modulation = float(
        np.hypot(coeff_recovered[1], coeff_recovered[2])
        / max(abs(coeff_recovered[0]), 1e-20)
    )
    modulation_error = abs(recovered_modulation - target_modulation) / max(
        target_modulation, 1e-20
    )
    recovered_phase = math.degrees(math.atan2(-coeff_recovered[2], coeff_recovered[1]))
    phase_error = abs(_wrapped_angle_deg(recovered_phase - phase_deg))
    passes = (
        correlation >= thresholds.correlation
        and leakage <= thresholds.leakage
        and modulation_error <= thresholds.modulation_error
    )
    return PatternQuality(
        correlation=correlation,
        leakage=leakage,
        modulation_error=modulation_error,
        fitted_phase_error_deg=phase_error,
        passes=bool(passes),
    )


def simulate_from_phase(
    slm_phase_low: np.ndarray,
    target_low: np.ndarray,
    target_frequency: float,
    angle_deg: float,
    phase_deg: float,
    carrier_frequency: float,
    iris_radius_mm: float,
    cfg: PhysicalBandwidthConfig,
) -> RecoveredPattern:
    phase_low = np.asarray(slm_phase_low, dtype=np.float64)
    if phase_low.shape != (cfg.slm_pixels, cfg.slm_pixels):
        raise ValueError("SLM phase shape does not match configuration")
    incident_low = gaussian_amplitude(
        cfg.slm_pixels, cfg.pixel_pitch_m, cfg.gaussian_radius_m
    )
    active_field = hold_pixels(incident_low * np.exp(1j * phase_low), cfg.oversample)
    phase = hold_pixels(phase_low, cfg.oversample)
    spectrum, iris, selected, recovered, diagnostics = _propagate_phase_field(
        active_field, carrier_frequency, iris_radius_mm, cfg
    )
    target_up = target_intensity(
        cfg.slm_pixels * cfg.oversample,
        target_frequency,
        angle_deg,
        phase_deg,
    )
    recovered_intensity = np.abs(recovered) ** 2
    quality = sinusoid_quality(
        target_up,
        recovered_intensity,
        target_frequency,
        angle_deg,
        phase_deg,
        cfg.thresholds,
    )
    return RecoveredPattern(
        target_intensity=target_up,
        slm_phase=phase,
        full_spectrum=spectrum,
        iris=iris,
        selected_spectrum=selected,
        recovered_field=recovered,
        recovered_intensity=recovered_intensity,
        quality=quality,
        diagnostics=diagnostics,
    )


def simulate_recovered_pattern(
    target_frequency: float,
    angle_deg: float,
    phase_deg: float,
    carrier_frequency: float,
    iris_radius_mm: float,
    cfg: PhysicalBandwidthConfig,
) -> RecoveredPattern:
    target_low = target_intensity(
        cfg.slm_pixels, target_frequency, angle_deg, phase_deg
    )
    active_field, phase_up = _phase_only_slm_field(
        target_low, carrier_frequency, cfg
    )
    spectrum, iris, selected, recovered, diagnostics = _propagate_phase_field(
        active_field, carrier_frequency, iris_radius_mm, cfg
    )
    target_up = target_intensity(
        cfg.slm_pixels * cfg.oversample,
        target_frequency,
        angle_deg,
        phase_deg,
    )
    recovered_intensity = np.abs(recovered) ** 2
    quality = sinusoid_quality(
        target_up,
        recovered_intensity,
        target_frequency,
        angle_deg,
        phase_deg,
        cfg.thresholds,
    )
    return RecoveredPattern(
        target_intensity=target_up,
        slm_phase=phase_up,
        full_spectrum=spectrum,
        iris=iris,
        selected_spectrum=selected,
        recovered_field=recovered,
        recovered_intensity=recovered_intensity,
        quality=quality,
        diagnostics=diagnostics,
    )
