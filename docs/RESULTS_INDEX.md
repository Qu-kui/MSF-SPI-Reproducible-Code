# Reference results index

## Figure R5: physically constrained bandwidth comparison

- Figure: `results/reference/physical_comparison/Figure_R5.png`
- Content: original USAF-1951 target, ideal circularly band-limited reference, conventional SPI, and MSF-SPI.
- Important scope: this archived figure uses the earlier 64 x 64 physical study with radii 7 and 14 cycles/FOV. It is not the 256 x 256 production robustness configuration.
- Related command: `scripts/reproduce_physical_comparison.py`.

## Figure R6: final calibrated robustness analysis

- Figure: `results/reference/robustness/Figure_R6.png`
- Metrics: `results/reference/robustness/summary_metrics.json`
- Configuration: `results/reference/robustness/configuration.json`
- Content: six independent perturbation sweeps reconstructed with the final sparse system-response correction. Stochastic points report three fixed realizations.
- Related command: `scripts/reproduce_robustness.py`.

## Angular quantization comparison

- Figure: `results/reference/angular_quantization/comparison.png`
- Summary: `results/reference/angular_quantization/summary.json`
- Content: 0.001-degree and 0.5-degree acquisitions. Both use the response matrix calibrated at 0.001 degrees, so the coarse result includes calibration-transfer mismatch.
- Related command: `scripts/reproduce_ideal_reconstruction.py` and the archived ideal calibration files.

All compact files are covered by `data/precomputed/checksums.sha256`.
