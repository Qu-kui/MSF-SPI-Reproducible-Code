from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .config import PhysicalSLMConfig as PhysicalBandwidthConfig
from .order_model import normalize_phase_sequence, simulate_recovered_pattern


PHASES = (0.0, 90.0, 180.0, 270.0)
PATTERN_ENCODING_VERSION = 3


class PhysicalQuartetCache:
    """Generate and persist normalized four-phase physical SLM intensities."""

    def __init__(
        self,
        cfg: PhysicalBandwidthConfig,
        *,
        carrier: float,
        iris_mm: float,
        root: Path,
    ) -> None:
        self.cfg = cfg
        self.carrier = float(carrier)
        self.iris_mm = float(iris_mm)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._memory: dict[tuple[float, float], np.ndarray] = {}
        self._generated_count = 0

    def _key(self, vector: np.ndarray) -> tuple[float, float]:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (2,):
            raise ValueError("frequency vector must contain qx and qy")
        return tuple(np.round(values, 8))

    def _path(self, key: tuple[float, float]) -> Path:
        payload = {
            "vector": key,
            "carrier": self.carrier,
            "iris_mm": self.iris_mm,
            "slm_pixels": self.cfg.slm_pixels,
            "oversample": self.cfg.oversample,
            "fft_size": self.cfg.fft_size,
            "wavelength_m": self.cfg.wavelength_m,
            "corner_intensity": self.cfg.corner_intensity,
            "fill_factor": self.cfg.fill_factor,
            "encoding_version": PATTERN_ENCODING_VERSION,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
        return self.root / f"quartet_{digest}.npy"

    def get(self, vector: np.ndarray) -> np.ndarray:
        key = self._key(vector)
        if key in self._memory:
            return self._memory[key]
        size = self.cfg.slm_pixels * self.cfg.oversample
        path = self._path(key)
        if path.exists():
            cached = np.load(path)
            if cached.shape == (4, size, size):
                self._memory[key] = cached
                return cached
        frequency = float(np.linalg.norm(key))
        angle = math.degrees(math.atan2(key[1], key[0])) if frequency else 0.0
        simulations = [
            simulate_recovered_pattern(
                frequency,
                angle,
                phase,
                self.carrier,
                self.iris_mm,
                self.cfg,
            )
            for phase in PHASES
        ]
        quartet = normalize_phase_sequence(
            np.stack([item.recovered_intensity for item in simulations])
        ).astype(np.float32)
        np.save(path, quartet)
        self._memory[key] = quartet
        self._generated_count += 1
        return quartet

    @property
    def unique_vector_count(self) -> int:
        return len(self._memory)

    @property
    def generated_count(self) -> int:
        return self._generated_count
