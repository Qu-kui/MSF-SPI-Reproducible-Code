import numpy as np
import pytest

from msf_spi.physical.config import PhysicalSLMConfig
from msf_spi.physical.order_model import simulate_recovered_pattern
from msf_spi.physical.propagation import gaussian_amplitude


def test_gaussian_corner_intensity_matches_configuration():
    config = PhysicalSLMConfig.quick()
    amplitude = gaussian_amplitude(
        config.slm_pixels,
        config.pixel_pitch_m,
        config.gaussian_radius_m,
    )
    intensity = amplitude**2
    assert intensity[0, 0] / intensity.max() == pytest.approx(
        config.corner_intensity, rel=3e-2
    )


def test_fill_factor_changes_pixel_aperture_response():
    full_config = PhysicalSLMConfig.quick(fill_factor=1.0)
    finite_config = PhysicalSLMConfig.quick(fill_factor=0.95)
    full = simulate_recovered_pattern(1.0, 0.0, 0.0, 3.0, 0.5, full_config)
    finite = simulate_recovered_pattern(1.0, 0.0, 0.0, 3.0, 0.5, finite_config)
    assert full.recovered_intensity.shape == (32, 32)
    assert not np.allclose(full.recovered_intensity, finite.recovered_intensity)


def test_recovered_pattern_reports_finite_quality_metrics():
    config = PhysicalSLMConfig.quick()
    result = simulate_recovered_pattern(1.0, 45.0, 90.0, 3.0, 0.5, config)
    assert np.isfinite(result.quality.correlation)
    assert np.isfinite(result.quality.leakage)
    assert np.isfinite(result.quality.modulation_error)

