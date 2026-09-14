# MSF-SPI Reproducible Simulation Code

This repository contains the numerical models, calibration routines, and compact reference artifacts used to study Moire sum-frequency single-pixel imaging (MSF-SPI). It supports a simulation-based proof of concept; it does not claim experimental validation or measured hardware performance.

## What is included

- An idealized algebraic model with a 64 x 64 programmable SLM grid, exact nearest-neighbor pixel replication to 512 x 512, a reference-image-derived angular-difference library, and an extended circular Fourier radius from 32 to 64 cycles/FOV.
- A pixel-resolved wave-optics model of a phase-only SLM, Gaussian illumination, finite pixel pitch and fill factor, blazed-carrier encoding, positive-first-order spatial filtering, finite relay pupils, and a unit-magnification relay.
- Global five-term Eq. (10) calibration and object-independent sparse system-response calibration.
- A six-factor robustness study for mask angle, mask frequency, SLM static phase, bucket-detector noise, calibration mismatch, and conjugate-plane defocus.
- Exact compact inputs, precomputed calibration matrices, reference figures, metrics, and SHA-256 checksums.

## Repository map

```text
configs/                  Paper-scale and quick configurations
data/                     Target, angle library, and calibrated responses
docs/                     Assumptions, calibration, provenance, and commands
results/reference/        Compact figures and metrics used in the response
scripts/                  Public command-line entry points
src/msf_spi/              Installable scientific Python package
tests/                    Unit, CLI, documentation, and release checks
tools/                    Provenance and checksum utilities
```

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The released workflows run on CPU. Optional CUDA support can be installed with `python -m pip install -e ".[cuda]"`, but CUDA is not required by the public commands or tests.

## Five-minute validation

The reduced validation executes all four public workflows using a small CPU configuration:

```bash
python scripts/run_quick_validation.py --output results/generated/quick_validation
```

Success is recorded as `"status": "passed"` in `results/generated/quick_validation/validation_report.json`.

## Full workflows

Idealized algebraic reconstruction:

```bash
python scripts/reproduce_ideal_reconstruction.py --config configs/ideal_model.json --angle-library data/angle_library/optimal_angle_differences.json --output results/generated/ideal
```

Pixel-resolved SLM bandwidth evaluation:

```bash
python scripts/evaluate_slm_bandwidth.py --config configs/physical_slm_256.json --output results/generated/bandwidth
```

Physical SPI/MSF-SPI comparison:

```bash
python scripts/reproduce_physical_comparison.py --config configs/physical_slm_256.json --response-matrix data/precomputed/physical_256_response_matrix.npz --output results/generated/physical_comparison
```

Final sparse-response-corrected robustness analysis:

```bash
python scripts/reproduce_robustness.py --response-matrix data/precomputed/physical_256_response_matrix.npz --output results/generated/robustness
```

Paper-scale physical runs are computationally expensive. They use a 256 x 256 SLM, a 1024 x 1024 sample-plane grid, a 2048 x 2048 Fourier grid, many four-phase frequency groups, and repeated stochastic sweeps. Runtime and memory depend strongly on CPU, RAM, cache reuse, and storage speed. Use the quick validation before starting a full run and retain the generated cache when resuming.

## Scientific interpretation

The idealized and physical models answer different questions and must not be mixed when reporting results. The idealized model tests the algebraic frequency-mixing and reconstruction pipeline under deliberately favorable optical assumptions. The physical model asks whether suitable SLM illumination can be formed after phase encoding, diffraction-order selection, and finite-aperture relay propagation.

The physical model uses collinear SLM and external-mask frequency vectors, corresponding to an angular difference of 0 degrees. It does not use the reference-image angle optimization because positive-first-order filtering produces approximately sinusoidal sample-plane illumination instead of the block-replicated pattern assumed by the idealized model. See [Model assumptions](docs/MODEL_ASSUMPTIONS.md) and [Calibration](docs/CALIBRATION.md).

The reported operational bandwidth of 32 cycles/FOV uses the declared engineering criteria correlation >= 0.95, residual spectral leakage <= 0.10, and modulation-depth error <= 0.10. All 144 orientation-phase cases pass at 32 cycles/FOV, whereas two cases fail at 32.5 cycles/FOV. The configuration also retains a substantially stricter diagnostic tier (0.99, 0.01, and 0.05, respectively); that tier is not the basis of the reported 32-cycles/FOV boundary.

## Reference artifacts and tolerances

[Results index](docs/RESULTS_INDEX.md) identifies the archived figures, metrics, configurations, and regeneration commands. Floating-point results can vary slightly with dependency versions and hardware. Compare trends and reported tolerances rather than requiring bitwise equality for newly generated floating-point arrays. Archived input and calibration files are verified byte-for-byte by SHA-256.

## Data and licensing

The software is provided under the MIT License. The included USAF-1951 image is the exact target used in the validated simulations and is included as a publicly available test resource for reproducibility. See [README.md](data/README.md) for further information.

If you use this repository, cite the associated MSF-SPI article and the software metadata in `CITATION.cff`.
