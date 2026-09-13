import json
from pathlib import Path

import pytest

from msf_spi.physical.config import PhysicalSLMConfig


ROOT = Path(__file__).resolve().parents[1]


def test_paper_configuration_matches_reported_physical_model():
    payload = json.loads((ROOT / "configs/physical_slm_256.json").read_text())
    config = PhysicalSLMConfig.from_mapping(payload)
    assert config.wavelength_m == pytest.approx(532e-9)
    assert config.slm_pixels == 256
    assert config.pixel_pitch_m == pytest.approx(8e-6)
    assert config.fill_factor == pytest.approx(0.95)
    assert config.corner_intensity == pytest.approx(0.80)
    assert config.carrier_cycles_per_fov == pytest.approx(100.0)
    assert config.first_order_iris_radius_mm == pytest.approx(1.75)
    assert config.conventional_radius == pytest.approx(32.0)
    assert config.msf_radius == pytest.approx(64.0)
    assert config.operational_thresholds.correlation == pytest.approx(0.95)
    assert config.operational_thresholds.leakage == pytest.approx(0.10)
    assert config.operational_thresholds.modulation_error == pytest.approx(0.10)
    assert config.thresholds.correlation == pytest.approx(0.99)


def test_invalid_fill_factor_is_rejected():
    with pytest.raises(ValueError, match="fill_factor"):
        PhysicalSLMConfig(fill_factor=1.1)
