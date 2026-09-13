# Reproducibility guide

Install the repository with `python -m pip install -e ".[dev]"`, then run `python scripts/run_quick_validation.py --output results/generated/quick_validation` from any directory. All entry points resolve defaults relative to the repository rather than the caller's current directory.

## Reviewer-response analyses

| Analysis | Command | Configuration or calibration |
|---|---|---|
| Ideal algebraic MSF-SPI | `python scripts/reproduce_ideal_reconstruction.py --config configs/ideal_model.json --output results/generated/ideal` | Reference-image angle library; 64 x 64 to 512 x 512 nearest-neighbor replication |
| SLM usable bandwidth | `python scripts/evaluate_slm_bandwidth.py --config configs/physical_slm_256.json --output results/generated/bandwidth` | 256 x 256 phase-only SLM, finite fill factor and pupils |
| Physical SPI/MSF-SPI comparison | `python scripts/reproduce_physical_comparison.py --config configs/physical_slm_256.json --response-matrix data/precomputed/physical_256_response_matrix.npz --output results/generated/physical_comparison` | Collinear frequency decomposition; angular difference 0 degrees |
| Six-error robustness | `python scripts/reproduce_robustness.py --response-matrix data/precomputed/physical_256_response_matrix.npz --output results/generated/robustness` | Frozen global weights and sparse response matrix |

The archived Figure R5 is an earlier compact physical comparison at a conventional radius of 7 cycles/FOV and an extended radius of 14 cycles/FOV. The released 256 x 256 production configuration instead uses radii 32 and 64 cycles/FOV for the expanded physical and robustness analyses. These results must be labeled separately.

Figure R6 and `results/reference/robustness/summary_metrics.json` correspond to the final sparse-response-corrected six-factor robustness algorithm. The archived angular-quantization comparison reuses the response matrix calibrated at 0.001 degrees for the 0.5-degree condition; it does not refit an angle-specific matrix.

The 256 x 256 bandwidth boundary was independently rerun with a fresh cache. Under the operational criteria (correlation >= 0.95, leakage <= 0.10, and modulation-depth error <= 0.10), all 144 orientation-phase cases pass at 32 cycles/FOV and 142 of 144 pass at 32.5 cycles/FOV. The stricter 0.99/0.01/0.05 tier is retained as a conservative diagnostic and must not be cited as the criterion that yields the 32-cycles/FOV boundary.

Run `python scripts/verify_release.py --quick --report results/generated/release_verification.json` before public deposition. Exact archived-file checksums must match. Newly computed floating-point metrics may differ slightly across numerical libraries and hardware.
