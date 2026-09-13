import json

import numpy as np

from msf_spi.ideal.patterns import (
    AngleLibraryEntry,
    equivalent_illumination_pattern,
    load_angle_library,
    quantize_angle,
)
from msf_spi.ideal.simulation import select_frequency_quartets


def test_angle_library_is_parsed_into_complete_phase_quartets(tmp_path):
    payload = {
        "optimal_angle_diffs": {
            "fx1_32_fx2_31.953_tfx_63_tfy_11_angle_9.904_p90_-90": 2.0,
            "fx1_32_fx2_31.953_tfx_63_tfy_11_angle_9.904_p90_0": 8.0,
            "fx1_32_fx2_31.953_tfx_63_tfy_11_angle_9.904_p90_90": 2.0,
            "fx1_32_fx2_31.953_tfx_63_tfy_11_angle_9.904_p90_180": 1.0,
        }
    }
    source = tmp_path / "angles.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    entries = load_angle_library(source)
    groups = select_frequency_quartets(entries)
    assert len(entries) == 4
    assert list(groups) == [(63, 11)]
    assert [round(row.total_phase_deg) for row in groups[(63, 11)]] == [0, 90, 180, 270]


def test_equivalent_pattern_preserves_pixel_replication_and_bounds():
    entry = AngleLibraryEntry(
        slm_frequency=32.0,
        mask_frequency=32.0,
        target_fx=40,
        target_fy=0,
        target_angle_deg=0.0,
        slm_phase_deg=90.0,
        mask_phase_deg=-90.0,
        angular_difference_deg=2.0,
    )
    pattern = equivalent_illumination_pattern(entry, size=64, slm_pixels=8)
    assert pattern.shape == (64, 64)
    assert np.isfinite(pattern).all()
    assert pattern.min() >= 0.0
    assert pattern.max() <= 1.0


def test_angle_quantization_is_explicit():
    assert quantize_angle(12.24, 0.5) == 12.0
    assert quantize_angle(12.26, 0.5) == 12.5

