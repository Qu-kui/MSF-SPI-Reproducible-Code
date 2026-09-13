import numpy as np

from msf_spi.physical.comparison import run_fmax_comparison
from msf_spi.physical.config import PhysicalSLMConfig


def test_quick_physical_comparison_returns_complete_result(tmp_path):
    config = PhysicalSLMConfig.quick()
    size = config.slm_pixels * config.oversample
    image = np.zeros((size, size), dtype=float)
    image[size // 4 : 3 * size // 4, size // 3 : 2 * size // 3] = 1.0
    result = run_fmax_comparison(
        image,
        config,
        fmax=config.conventional_radius,
        carrier=config.carrier_cycles_per_fov,
        iris_mm=config.first_order_iris_radius_mm,
        cache_dir=tmp_path,
        progress_every=0,
    )
    assert result["reference"].shape == (size, size)
    assert result["spi_reconstruction"].shape == (size, size)
    assert result["msf_sparse_reconstruction"].shape == (size, size)
    assert result["msf_frequency_count"] > result["spi_frequency_count"]
    assert "msf_sparse_vs_ideal_metrics" in result

