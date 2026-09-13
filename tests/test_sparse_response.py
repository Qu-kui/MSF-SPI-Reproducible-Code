import numpy as np
import pytest
from scipy import sparse

from msf_spi.calibration.sparse_response import (
    apply_real_linear_matrix,
    load_response_matrix,
    save_response_matrix,
    solve_calibrated_coefficients,
)


def test_sparse_response_round_trip(tmp_path):
    matrix = sparse.csr_matrix(np.array([[1.0, 0.1], [0.2, 0.9]]))
    path = tmp_path / "response.npz"
    save_response_matrix(path, matrix)
    loaded = load_response_matrix(path)
    assert np.array_equal(loaded.toarray(), matrix.toarray())


def test_sparse_solver_recovers_coupled_complex_coefficient():
    matrix = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.1, 0.2, 0.0],
                [0.0, 0.9, 0.1, 0.1],
                [0.1, 0.0, 1.1, -0.1],
                [0.0, 0.2, 0.0, 0.8],
            ]
        )
    )
    expected = np.array([1.0 + 2.0j, -0.5 + 0.25j])
    measured = apply_real_linear_matrix(matrix, expected)
    recovered = solve_calibrated_coefficients(matrix, measured)
    assert recovered == pytest.approx(expected, abs=1e-10)
