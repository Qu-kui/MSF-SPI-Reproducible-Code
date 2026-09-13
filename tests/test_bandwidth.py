from msf_spi.physical.bandwidth import evaluate_frequency_dual
from msf_spi.physical.config import PhysicalSLMConfig


def test_bandwidth_row_contains_declared_quality_metrics():
    config = PhysicalSLMConfig.quick()
    rows = evaluate_frequency_dual(
        config,
        target_frequency=1.0,
        carrier_frequency=3.0,
        iris_radius_mm=0.5,
        angles_deg=(0.0,),
        phases_deg=(0.0,),
        stage="test",
    )
    assert len(rows) == 1
    assert {
        "target_frequency",
        "angle_deg",
        "phase_deg",
        "correlation",
        "leakage",
        "modulation_error",
        "strict_passes",
        "operational_passes",
    } <= rows[0].keys()


def test_production_thresholds_match_the_final_code():
    config = PhysicalSLMConfig.production_256()
    assert config.thresholds.correlation == 0.99
    assert config.thresholds.leakage == 0.01
    assert config.thresholds.modulation_error == 0.05

