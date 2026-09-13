import numpy as np
import pytest

from msf_spi.physical.frequency_mixing import parallel_frequency_decomposition


def test_physical_frequency_decomposition_is_collinear():
    result = parallel_frequency_decomposition(np.array([30.0, 40.0]), 32.0)
    mask_cross = result["mask"][0] * result["target"][1] - result["mask"][1] * result["target"][0]
    slm_cross = result["slm"][0] * result["target"][1] - result["slm"][1] * result["target"][0]
    assert mask_cross == pytest.approx(0.0)
    assert slm_cross == pytest.approx(0.0)
    assert result["slm"] + result["mask"] == pytest.approx(result["target"])


def test_in_band_target_uses_only_the_slm():
    result = parallel_frequency_decomposition(np.array([3.0, 4.0]), 7.0)
    assert result["mask"] == pytest.approx(np.zeros(2))
    assert result["slm"] == pytest.approx(np.array([3.0, 4.0]))


def test_target_beyond_extended_radius_is_rejected():
    with pytest.raises(ValueError, match="twice"):
        parallel_frequency_decomposition(np.array([65.0, 0.0]), 32.0)
