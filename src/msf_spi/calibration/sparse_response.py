from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


@dataclass(frozen=True)
class FrozenResponse:
    dc: complex
    gain: complex
    conjugate_gain: complex


def four_step_kernel(patterns: np.ndarray) -> np.ndarray:
    values = np.asarray(patterns)
    if values.ndim != 3 or values.shape[0] != 4:
        raise ValueError("four phase patterns with shape (4, y, x) are required")
    return (values[0] - values[2]) + 1j * (values[1] - values[3])


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
    return FrozenResponse(
        dc=dc,
        gain=complex(np.vdot(target, zero_mean) / sample_count),
        conjugate_gain=complex(np.vdot(np.conj(target), zero_mean) / sample_count),
    )


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


def _basis_responses_from_spectrum(
    spectrum: np.ndarray,
    point: tuple[int, int],
    response: FrozenResponse,
) -> tuple[complex, complex]:
    size = spectrum.shape[0]
    qx, qy = point
    positive = spectrum[qy % size, qx % size]
    negative = spectrum[-qy % size, -qx % size]
    offset = 0.5 - size / 2.0
    phase = np.exp(2j * np.pi * (qx + qy) * offset / size)
    cosine_sum = 0.5 * (phase * negative + np.conj(phase) * positive)
    sine_sum = (phase * negative - np.conj(phase) * positive) / (2j)
    raw_real = (2.0 / (size * size)) * cosine_sum
    raw_imag = (-2.0 / (size * size)) * sine_sum
    return (
        apply_frozen_response(raw_real, response, 0.0),
        apply_frozen_response(raw_imag, response, 0.0),
    )


def matrix_from_kernels(
    kernels: Iterable[np.ndarray],
    points: Sequence[tuple[int, int]],
    responses: Sequence[FrozenResponse] | None = None,
    *,
    max_terms: int = 96,
    relative_threshold: float = 5e-4,
) -> sparse.csr_matrix:
    point_list = [(int(qx), int(qy)) for qx, qy in points]
    response_list = None if responses is None else list(responses)
    if response_list is not None and len(response_list) != len(point_list):
        raise ValueError("response and point counts do not match")
    if max_terms < 1:
        raise ValueError("max_terms must be positive")
    qx = np.asarray([point[0] for point in point_list], dtype=int)
    qy = np.asarray([point[1] for point in point_list], dtype=int)
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    count = 0
    for row_index, kernel in enumerate(kernels):
        if row_index >= len(point_list):
            raise ValueError("kernel and point counts do not match")
        values = np.asarray(kernel, dtype=np.complex128)
        if values.ndim != 2 or values.shape[0] != values.shape[1]:
            raise ValueError("each kernel must be square")
        response = (
            response_list[row_index]
            if response_list is not None
            else response_from_kernel(values, *point_list[row_index])
        )
        spectrum = np.fft.fft2(values)
        magnitudes = np.maximum(
            np.abs(spectrum[qy % values.shape[0], qx % values.shape[1]]),
            np.abs(spectrum[-qy % values.shape[0], -qx % values.shape[1]]),
        )
        main = float(magnitudes[row_index])
        threshold = max(main * float(relative_threshold), 1e-12)
        candidates = np.flatnonzero(magnitudes >= threshold)
        if candidates.size > max_terms:
            order = np.argpartition(magnitudes[candidates], -max_terms)[-max_terms:]
            candidates = candidates[order]
        if row_index not in candidates:
            candidates = np.append(candidates, row_index)
        for column_index in np.unique(candidates):
            response_a, response_b = _basis_responses_from_spectrum(
                spectrum,
                point_list[int(column_index)],
                response,
            )
            block = (
                (response_a.real, response_b.real),
                (response_a.imag, response_b.imag),
            )
            for local_row in range(2):
                for local_column in range(2):
                    value = float(block[local_row][local_column])
                    if abs(value) > 1e-12:
                        rows.append(2 * row_index + local_row)
                        columns.append(2 * int(column_index) + local_column)
                        data.append(value)
        count += 1
    if count != len(point_list):
        raise ValueError("kernel and point counts do not match")
    return sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(2 * len(point_list), 2 * len(point_list)),
    ).tocsr()


def matrix_from_phase_quartets(
    quartets: Iterable[np.ndarray],
    points: Sequence[tuple[int, int]],
    *,
    max_terms: int = 96,
    relative_threshold: float = 5e-4,
) -> sparse.csr_matrix:
    point_list = [(int(qx), int(qy)) for qx, qy in points]
    if len(set(point_list)) != len(point_list):
        raise ValueError("frequency points must be unique")
    return matrix_from_kernels(
        (four_step_kernel(quartet) for quartet in quartets),
        point_list,
        max_terms=max_terms,
        relative_threshold=relative_threshold,
    )


def apply_real_linear_matrix(
    matrix: sparse.spmatrix | np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.complex128)
    if values.ndim != 1:
        raise ValueError("coefficients must be one-dimensional")
    if matrix.shape != (2 * values.size, 2 * values.size):
        raise ValueError("matrix and coefficient dimensions do not match")
    packed = np.empty(2 * values.size, dtype=np.float64)
    packed[0::2] = values.real
    packed[1::2] = values.imag
    output = np.asarray(matrix @ packed).reshape(-1)
    return output[0::2] + 1j * output[1::2]


def solve_calibrated_coefficients(
    matrix: sparse.spmatrix,
    measured: np.ndarray,
    *,
    regularization: float = 0.0,
) -> np.ndarray:
    values = np.asarray(measured, dtype=np.complex128)
    one_case = values.ndim == 1
    if one_case:
        values = values[None, :]
    if values.ndim != 2 or matrix.shape != (2 * values.shape[1], 2 * values.shape[1]):
        raise ValueError("mixing matrix and measured coefficients are incompatible")
    rhs = np.empty((2 * values.shape[1], values.shape[0]), dtype=np.float64)
    rhs[0::2] = values.real.T
    rhs[1::2] = values.imag.T
    operator = sparse.csc_matrix(matrix)
    if regularization > 0.0:
        identity = sparse.eye(operator.shape[1], format="csc")
        normal = operator.T @ operator + float(regularization) * identity
        solved = sparse_linalg.spsolve(normal, operator.T @ rhs)
    else:
        solved = sparse_linalg.splu(operator).solve(rhs)
    if solved.ndim == 1:
        solved = solved[:, None]
    output = solved[0::2].T + 1j * solved[1::2].T
    return output[0] if one_case else output


class CalibratedResponseSolver:
    def __init__(self, matrix: sparse.spmatrix, regularization: float = 0.0) -> None:
        self.matrix = sparse.csr_matrix(matrix)
        self.regularization = float(regularization)
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("mixing matrix must be square")

    def solve(self, measured: np.ndarray) -> np.ndarray:
        return solve_calibrated_coefficients(
            self.matrix,
            measured,
            regularization=self.regularization,
        )


def save_response_matrix(path: str | Path, matrix: sparse.spmatrix) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(destination, sparse.csr_matrix(matrix))


def load_response_matrix(path: str | Path) -> sparse.csr_matrix:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"response matrix not found: {source}")
    return sparse.load_npz(source).tocsr()

