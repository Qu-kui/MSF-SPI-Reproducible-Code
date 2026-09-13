from __future__ import annotations

import numpy as np


def parallel_frequency_decomposition(
    target: np.ndarray,
    conventional_radius: float,
) -> dict[str, np.ndarray]:
    """Decompose a target vector into collinear SLM and external-mask vectors.

    This is the decomposition used by the pixel-resolved physical validation.
    Its SLM and mask vectors are parallel to the target, so their angular
    difference is zero degrees. Targets inside the conventional radius use the
    SLM alone.
    """
    vector = np.asarray(target, dtype=np.float64)
    if vector.shape != (2,):
        raise ValueError("target must be a two-component frequency vector")
    radius = float(conventional_radius)
    if radius <= 0.0:
        raise ValueError("conventional_radius must be positive")
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= radius + 1e-12:
        mask = np.zeros(2, dtype=np.float64)
        slm = vector.copy()
    else:
        if magnitude > 2.0 * radius + 1e-12:
            raise ValueError("target frequency exceeds twice the conventional radius")
        mask = radius * vector / magnitude
        slm = vector - mask
    return {
        "target": vector,
        "slm": slm,
        "mask": mask,
        "difference": slm - mask,
    }

