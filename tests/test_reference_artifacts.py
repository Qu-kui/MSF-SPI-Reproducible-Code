import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]


def test_reference_inputs_and_calibrations_are_loadable():
    target = cv2.imdecode(
        np.fromfile(ROOT / "data" / "targets" / "USAF-1951.jpg", dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    angles = json.loads(
        (ROOT / "data" / "angle_library" / "optimal_angle_differences.json").read_text(
            encoding="utf-8"
        )
    )["optimal_angle_diffs"]
    weights = json.loads(
        (ROOT / "data" / "precomputed" / "physical_256_global_weights.json").read_text(
            encoding="utf-8"
        )
    )["direct_weights"]
    matrix = sparse.load_npz(
        ROOT / "data" / "precomputed" / "physical_256_response_matrix.npz"
    )
    assert target is not None and target.size > 0
    assert len(angles) == 51408
    assert len(weights) == 5 and np.isfinite(weights).all()
    assert matrix.shape[0] == matrix.shape[1] and np.isfinite(matrix.data).all()


def test_all_published_data_checksums_match():
    checksum_file = ROOT / "data" / "precomputed" / "checksums.sha256"
    for line in checksum_file.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        content = (ROOT / relative).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected


def test_reference_metrics_describe_the_final_sparse_calibration():
    summary = json.loads(
        (ROOT / "results" / "reference" / "robustness" / "summary_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(summary) == {
        "mask_angle",
        "mask_frequency",
        "slm_static_phase",
        "detector_noise",
        "calibration_mismatch",
        "conjugate_defocus",
    }
    assert all(summary[name] for name in summary)
