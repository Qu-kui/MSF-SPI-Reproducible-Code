from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TypeAlias

import numpy as np


FEATURE_NAMES = ("moire", "slm", "mask", "difference", "uniform")
THEORETICAL_WEIGHTS = np.array([4.0, -2.0, -2.0, -1.0, 2.0], dtype=np.float64)

RecordSource: TypeAlias = (
    Iterable[tuple[np.ndarray, np.ndarray]]
    | Callable[[], Iterator[tuple[np.ndarray, np.ndarray]]]
)


def build_features(
    slm: np.ndarray,
    difference: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Build the five measurable intensity classes used by Eq. (10)."""
    slm_values = np.asarray(slm, dtype=np.float64)
    difference_values = np.asarray(difference, dtype=np.float64)
    mask_values = np.asarray(mask, dtype=np.float64)
    if slm_values.shape != difference_values.shape or slm_values.shape != mask_values.shape:
        raise ValueError("SLM, difference, and mask patterns must have matching shapes")
    return np.stack(
        (
            slm_values * mask_values,
            slm_values,
            mask_values,
            difference_values,
            np.ones_like(mask_values),
        ),
        axis=-1,
    )


def direct_weights_to_eq10(weights: np.ndarray) -> np.ndarray:
    """Convert direct linear weights to k_sum, k_SLM, k_mask, k_diff, k_white."""
    coefficients = np.asarray(weights, dtype=np.float64)
    if coefficients.shape != (5,) or not np.all(np.isfinite(coefficients)):
        raise ValueError("direct weights must contain five finite values")
    if abs(coefficients[0]) < 1e-12:
        raise ValueError("the Moire coefficient is too small for Eq. (10) conversion")
    k_sum = 4.0 / coefficients[0]
    return np.array(
        [
            k_sum,
            -coefficients[1] * k_sum / 2.0,
            -coefficients[2] * k_sum / 2.0,
            -coefficients[3] * k_sum,
            coefficients[4] * k_sum / 2.0,
        ],
        dtype=np.float64,
    )


def _records(source: RecordSource):
    return source() if callable(source) else source


def _normal_equations(
    source: RecordSource,
    coefficients: np.ndarray | None,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((5, 5), dtype=np.float64)
    vector = np.zeros(5, dtype=np.float64)
    for features, target in _records(source):
        design = np.asarray(features, dtype=np.float64).reshape(-1, 5)
        expected = np.asarray(target, dtype=np.float64).reshape(-1)
        if design.shape[0] != expected.size:
            raise ValueError("feature and target sample counts do not match")
        if coefficients is None:
            matrix += design.T @ design
            vector += design.T @ expected
        else:
            residual = design @ coefficients - expected
            weights = 1.0 / np.maximum(np.abs(residual), epsilon)
            matrix += design.T @ (weights[:, None] * design)
            vector += design.T @ (weights * expected)
    return matrix, vector


def _solve(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, vector, rcond=None)[0]


def fit_global_weights(
    records: RecordSource,
    *,
    iterations: int = 6,
    epsilon: float = 1e-6,
    eps: float | None = None,
    loss: str = "l1",
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Fit one object-independent set of global channel weights."""
    if loss not in {"l1", "l2"}:
        raise ValueError("loss must be 'l1' or 'l2'")
    if eps is not None:
        epsilon = float(eps)
    matrix, vector = _normal_equations(records, None, epsilon)
    coefficients = _solve(matrix, vector)
    completed = 1
    if loss == "l1":
        for _ in range(iterations):
            matrix, vector = _normal_equations(records, coefficients, epsilon)
            coefficients = _solve(matrix, vector)
            completed += 1
    return coefficients, {
        "loss": loss,
        "irls_iterations": int(iterations if loss == "l1" else 0),
        "linear_solves": completed,
        "condition_number": float(np.linalg.cond(matrix)),
    }
