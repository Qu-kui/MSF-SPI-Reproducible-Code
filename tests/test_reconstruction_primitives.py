import numpy as np

from msf_spi.metrics import image_metrics, normalize_image
from msf_spi.reconstruction.fourier import (
    four_step_coefficient,
    half_plane_frequency_points,
    reconstruct_from_half_plane,
)


def test_four_step_demodulation_recovers_complex_coefficient():
    buckets = np.array([8.0, 5.0, 2.0, 1.0])
    assert four_step_coefficient(buckets) == 6.0 + 4.0j


def test_frequency_points_are_unique_half_plane_samples():
    points = half_plane_frequency_points(2.0)
    assert (1, 0) in points
    assert (-1, 0) not in points
    assert len(points) == len(set(points))


def test_normalization_and_metrics_are_finite():
    reference = np.eye(8)
    reconstruction = reconstruct_from_half_plane(
        {(1, 0): 1.0 + 0.0j}, size=8, dc=reference.mean()
    )
    metrics = image_metrics(normalize_image(reference), normalize_image(reconstruction))
    assert np.isfinite(metrics["ssim"])
    assert np.isfinite(metrics["psnr_db"])
