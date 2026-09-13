from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class QualityThresholds:
    correlation: float = 0.99
    leakage: float = 0.01
    modulation_error: float = 0.05

    def validate(self) -> None:
        if not 0.0 <= self.correlation <= 1.0:
            raise ValueError("correlation threshold must lie in [0, 1]")
        if not 0.0 <= self.leakage <= 1.0:
            raise ValueError("leakage threshold must lie in [0, 1]")
        if self.modulation_error < 0.0:
            raise ValueError("modulation_error threshold must be non-negative")


@dataclass(frozen=True)
class PhysicalSLMConfig:
    wavelength_m: float = 532e-9
    slm_pixels: int = 256
    pixel_pitch_m: float = 8e-6
    oversample: int = 4
    fft_size: int = 2048
    order_lens_focal_m: float = 0.200
    order_lens_diameter_m: float = 0.0254
    relay_focal_m: float = 0.100
    relay_lens_diameter_m: float = 0.0254
    corner_intensity: float = 0.80
    fill_factor: float = 0.95
    phases_deg: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    coarse_carrier_values: tuple[float, ...] = (100.0,)
    coarse_iris_values_mm: tuple[float, ...] = (1.75,)
    coarse_target_values: tuple[float, ...] = (
        16.0,
        20.0,
        24.0,
        28.0,
        32.0,
        36.0,
        40.0,
    )
    coarse_angle_values_deg: tuple[float, ...] = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)
    final_angle_values_deg: tuple[float, ...] = tuple(np.arange(0.0, 180.0, 5.0))
    operational_thresholds: QualityThresholds = field(
        default_factory=lambda: QualityThresholds(0.95, 0.10, 0.10)
    )
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    boundary_step: float = 0.5
    conventional_radius: float = 32.0
    msf_radius: float = 64.0

    def __post_init__(self) -> None:
        self.validate()

    @property
    def active_width_m(self) -> float:
        return self.slm_pixels * self.pixel_pitch_m

    @property
    def gaussian_radius_m(self) -> float:
        return self.active_width_m / math.sqrt(-math.log(self.corner_intensity))

    @property
    def fourier_plane_m_per_cycle(self) -> float:
        return self.wavelength_m * self.order_lens_focal_m / self.active_width_m

    @property
    def order_lens_na(self) -> float:
        return self.order_lens_diameter_m / (2.0 * self.order_lens_focal_m)

    @property
    def relay_na(self) -> float:
        return self.relay_lens_diameter_m / (2.0 * self.relay_focal_m)

    @property
    def carrier_cycles_per_fov(self) -> float:
        return float(self.coarse_carrier_values[0])

    @property
    def first_order_iris_radius_mm(self) -> float:
        return float(self.coarse_iris_values_mm[0])

    @property
    def coarse_carriers(self) -> np.ndarray:
        return np.asarray(self.coarse_carrier_values, dtype=float)

    @property
    def coarse_iris_mm(self) -> np.ndarray:
        return np.asarray(self.coarse_iris_values_mm, dtype=float)

    @property
    def coarse_targets(self) -> np.ndarray:
        return np.asarray(self.coarse_target_values, dtype=float)

    @property
    def coarse_angles_deg(self) -> np.ndarray:
        return np.asarray(self.coarse_angle_values_deg, dtype=float)

    @property
    def final_angles_deg(self) -> np.ndarray:
        return np.asarray(self.final_angle_values_deg, dtype=float)

    def validate(self) -> None:
        if self.wavelength_m <= 0.0 or self.pixel_pitch_m <= 0.0:
            raise ValueError("wavelength_m and pixel_pitch_m must be positive")
        if self.slm_pixels < 8 or self.slm_pixels % 2:
            raise ValueError("slm_pixels must be an even integer greater than or equal to 8")
        if self.oversample < 1 or self.fft_size < self.slm_pixels * self.oversample:
            raise ValueError("FFT grid must contain the oversampled SLM aperture")
        if not 0.0 < self.corner_intensity < 1.0:
            raise ValueError("corner_intensity must lie in (0, 1)")
        if not 0.0 < self.fill_factor <= 1.0:
            raise ValueError("fill_factor must lie in (0, 1]")
        if not self.phases_deg:
            raise ValueError("at least one phase state is required")
        if not self.coarse_target_values or max(self.coarse_target_values) >= self.slm_pixels / 2:
            raise ValueError("target scan must remain below the SLM Nyquist limit")
        if self.conventional_radius <= 0.0 or self.msf_radius < self.conventional_radius:
            raise ValueError("frequency radii are inconsistent")
        self.operational_thresholds.validate()
        self.thresholds.validate()

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PhysicalSLMConfig":
        payload = dict(values)
        for name in ("operational_thresholds", "thresholds"):
            if isinstance(payload.get(name), Mapping):
                payload[name] = QualityThresholds(**payload[name])
        tuple_fields = (
            "phases_deg",
            "coarse_carrier_values",
            "coarse_iris_values_mm",
            "coarse_target_values",
            "coarse_angle_values_deg",
            "final_angle_values_deg",
        )
        for name in tuple_fields:
            if name in payload:
                payload[name] = tuple(float(value) for value in payload[name])
        return cls(**payload)

    @classmethod
    def production(cls, **overrides: Any) -> "PhysicalSLMConfig":
        return replace(cls(), **overrides)

    @classmethod
    def production_256(cls, **overrides: Any) -> "PhysicalSLMConfig":
        return cls.production(**overrides)

    @classmethod
    def quick(cls, **overrides: Any) -> "PhysicalSLMConfig":
        config = cls(
            slm_pixels=16,
            oversample=2,
            fft_size=128,
            fill_factor=0.95,
            coarse_carrier_values=(3.0,),
            coarse_iris_values_mm=(0.5,),
            coarse_target_values=(1.0, 2.0, 3.0),
            coarse_angle_values_deg=(0.0, 45.0, 90.0, 135.0),
            final_angle_values_deg=(0.0, 45.0, 90.0, 135.0),
            operational_thresholds=QualityThresholds(0.95, 0.10, 0.10),
            thresholds=QualityThresholds(0.95, 0.10, 0.10),
            conventional_radius=2.0,
            msf_radius=4.0,
        )
        return replace(config, **overrides)


PhysicalBandwidthConfig = PhysicalSLMConfig
