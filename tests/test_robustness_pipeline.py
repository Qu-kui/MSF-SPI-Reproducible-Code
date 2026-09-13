from msf_spi.robustness.pipeline import run_robustness_pipeline


def test_quick_pipeline_uses_one_frozen_sparse_response(tmp_path):
    result = run_robustness_pipeline(tmp_path, quick=True, resume=False)
    assert set(result["family_results"]) == {
        "mask_angle",
        "mask_frequency",
        "slm_static_phase",
        "detector_noise",
        "calibration_mismatch",
        "conjugate_defocus",
    }
    assert result["response_matrix"]["shape"][0] == result["response_matrix"]["shape"][1]
    assert result["baseline"]["reconstruction"].shape == result["reference"].shape

