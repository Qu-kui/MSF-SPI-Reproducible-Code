import numpy as np
import pytest

from msf_spi.calibration.global_weights import (
    THEORETICAL_WEIGHTS,
    build_features,
    direct_weights_to_eq10,
    fit_global_weights,
)


def test_eq10_identity_for_nonnegative_intensity_patterns():
    a = np.linspace(0.0, 2.0 * np.pi, 31)
    b = np.linspace(0.3, 2.0 * np.pi + 0.3, 31)
    slm = (1.0 + np.cos(a)) / 2.0
    mask = (1.0 + np.cos(b)) / 2.0
    difference = (1.0 + np.cos(a - b)) / 2.0
    sum_pattern = (1.0 + np.cos(a + b)) / 2.0
    recovered = 4.0 * slm * mask - 2.0 * slm - 2.0 * mask - difference + 2.0
    assert recovered == pytest.approx(sum_pattern)


def test_global_weight_fit_recovers_known_coefficients():
    rng = np.random.default_rng(260603)
    features = rng.normal(size=(1000, 5))
    target = features @ THEORETICAL_WEIGHTS
    fitted, diagnostics = fit_global_weights([(features, target)], loss="l2")
    assert fitted == pytest.approx(THEORETICAL_WEIGHTS, abs=1e-10)
    assert diagnostics["linear_solves"] == 1


def test_feature_order_and_eq10_conversion():
    slm = np.array([[0.2, 0.8]])
    mask = np.array([[0.4, 0.6]])
    difference = np.array([[0.1, 0.9]])
    features = build_features(slm, difference, mask)
    assert features.shape == (1, 2, 5)
    assert direct_weights_to_eq10(THEORETICAL_WEIGHTS) == pytest.approx(np.ones(5))

