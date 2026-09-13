import numpy as np

from msf_spi.robustness.cases import build_robustness_cases, zero_mean_rms_map
from msf_spi.robustness.errors import calibration_drift, static_slm_phase_map


EXPECTED_FAMILIES = {
    "mask_angle",
    "mask_frequency",
    "slm_static_phase",
    "detector_noise",
    "calibration_mismatch",
    "conjugate_defocus",
}


def test_six_declared_robustness_families_are_present():
    assert set(build_robustness_cases()) == EXPECTED_FAMILIES


def test_random_realizations_are_deterministic():
    assert np.array_equal(
        static_slm_phase_map(16, 0.1, 260603),
        static_slm_phase_map(16, 0.1, 260603),
    )
    assert np.array_equal(
        calibration_drift(5.0, 260603),
        calibration_drift(5.0, 260603),
    )
    assert np.array_equal(
        zero_mean_rms_map(16, 0.1, 260603),
        zero_mean_rms_map(16, 0.1, 260603),
    )

