# MSF-SPI Reproducible Code Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an English-only, traceable, tested research-code repository that reproduces the final idealized MSF-SPI, physical-SLM bandwidth and reconstruction, calibration, sparse-response correction, and six robustness studies.

**Architecture:** Preserve the validated numerical kernels while moving them into an installable `msf_spi` package with separate `ideal`, `physical`, `calibration`, `reconstruction`, and `robustness` namespaces. Thin command-line scripts load explicit JSON configurations, write self-describing artifacts, and support both quick CPU validation and full paper-scale execution. Original working directories remain read-only; released files carry provenance records and checksums.

**Tech Stack:** Python 3.11+, NumPy, SciPy, OpenCV, Matplotlib, tqdm, psutil, pytest, optional CuPy/CUDA, JSON/NPZ artifacts, Git.

---

Repository root for every path below:

`D:/DeskTop/审稿意见/MSF-SPI-Reproducible-Code`

Source roots are read-only:

- `D:/DeskTop/审稿意见`
- `D:/DeskTop/我的工作/摩尔纹基底单像素成像-done/最终综合测试`

## File Map

### Repository metadata

- Create `README.md`: overview, installation, model distinction, quick start, full reproduction commands, expected resources, citation, and limitations.
- Create `LICENSE`: MIT licence with the neutral copyright holder “MSF-SPI authors”.
- Create `CITATION.cff`: project citation metadata without an invented DOI.
- Create `pyproject.toml`: package metadata, dependencies, optional CUDA and development extras, and pytest settings.
- Create `requirements.txt`: simple compatibility installation list.
- Create `.gitignore`: exclude caches, generated outputs, virtual environments, and large transient measurements.

### Configuration

- Create `configs/ideal_model.json`.
- Create `configs/physical_slm_256.json`.
- Create `configs/robustness_sweeps.json`.
- Create `configs/quick_validation.json`.

### Package

- Create `src/msf_spi/__init__.py` and `src/msf_spi/paths.py`.
- Create `src/msf_spi/metrics.py` from the validated shared metrics implementation.
- Create `src/msf_spi/ideal/full_simulation.py` as an English-only, functionally frozen copy of the final idealized source.
- Create `src/msf_spi/ideal/patterns.py`, `response.py`, and `dual_angle.py` from the final ideal-model helper modules.
- Create `src/msf_spi/physical/config.py`, `order_model.py`, `pattern_cache.py`, `bandwidth.py`, `frequency_mixing.py`, and `comparison.py`.
- Create `src/msf_spi/calibration/global_weights.py` and `sparse_response.py`.
- Create `src/msf_spi/reconstruction/fourier.py`.
- Create `src/msf_spi/robustness/cases.py`, `pipeline.py`, and `artifacts.py`.
- Add `__init__.py` files for every package namespace.

### Commands

- Create `scripts/reproduce_ideal_reconstruction.py`.
- Create `scripts/evaluate_slm_bandwidth.py`.
- Create `scripts/reproduce_physical_comparison.py`.
- Create `scripts/reproduce_robustness.py`.
- Create `scripts/run_quick_validation.py`.
- Create `scripts/verify_release.py`.

### Data and reference outputs

- Create `data/README.md`.
- Copy `data/targets/USAF-1951.jpg` from the exact test target used in the final runs.
- Copy `data/angle_library/optimal_angle_differences.json` from `Optimal_angle_diffs_SPI_circle_radial_stripes_optimized.json`.
- Copy `data/precomputed/physical_256_response_matrix.npz` from the validated physical robustness output.
- Create `data/precomputed/checksums.sha256`.
- Copy final compact configuration, metrics, and figures into `results/reference/physical_comparison` and `results/reference/robustness`.

### Documentation

- Create `docs/MODEL_ASSUMPTIONS.md`.
- Create `docs/CALIBRATION.md`.
- Create `docs/REPRODUCIBILITY.md`.
- Create `docs/FILE_PROVENANCE.md`.
- Create `docs/RESULTS_INDEX.md`.

### Tests

- Create focused tests under `tests/` for configuration, paths, Eq. (10), frequency enumeration and decomposition, physical propagation, global weights, sparse response, robustness determinism, CLI smoke execution, English-only content, private-path exclusion, and checksums.

---

### Task 1: Initialize the isolated release repository and source-integrity manifest

**Files:**

- Create: `.gitignore`
- Create: `tools/build_source_manifest.ps1`
- Create: `docs/SOURCE_INTEGRITY.md`
- Create: `tests/test_release_tree.py`

- [ ] **Step 1: Write the release-tree test**

Create `tests/test_release_tree.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_release_directories_exist():
    required = (
        "configs",
        "data",
        "docs",
        "results",
        "scripts",
        "src",
        "tests",
        "tools",
    )
    assert all((ROOT / name).is_dir() for name in required)


def test_private_source_roots_are_not_nested_in_release():
    names = {path.name for path in ROOT.rglob("*")}
    assert "最终综合测试" not in names
    assert "审稿意见" not in names
```

- [ ] **Step 2: Run the test and confirm it fails before scaffolding**

Run:

```powershell
python -m pytest tests/test_release_tree.py -v
```

Expected: failure because the release subdirectories do not yet exist.

- [ ] **Step 3: Create the approved directory tree and Git repository**

Run:

```powershell
git init -b main
New-Item -ItemType Directory -Force configs,data/targets,data/angle_library,data/precomputed,results/reference/physical_comparison,results/reference/robustness,scripts,src/msf_spi/ideal,src/msf_spi/physical,src/msf_spi/calibration,src/msf_spi/reconstruction,src/msf_spi/robustness,tests,tools | Out-Null
```

Expected: an empty Git repository and every approved directory.

- [ ] **Step 4: Add repository exclusions**

Create `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/
dist/
build/
results/generated/
**/cache/
**/masks_data/
**/measurement_batches/
*.tmp
```

- [ ] **Step 5: Add the source-manifest script**

Create `tools/build_source_manifest.ps1` with two explicit source roots, a fixed list of selected source files, SHA-256 hashing via `Get-FileHash`, and UTF-8 JSON output to `docs/source_manifest.json`. The script must only read source files and write the manifest inside the release repository.

The selected list must contain the exact inputs named in the approved design specification, the USAF target, the angle library, the final physical response matrix, and the final reference figures and JSON summaries.

- [ ] **Step 6: Generate and document the initial source hashes**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_source_manifest.ps1
```

Create `docs/SOURCE_INTEGRITY.md` explaining that source files are read-only, the manifest records their pre-packaging hashes, and a later verification compares them again.

- [ ] **Step 7: Run the release-tree test**

Run:

```powershell
python -m pytest tests/test_release_tree.py -v
```

Expected: two passing tests.

- [ ] **Step 8: Commit the scaffold**

```powershell
git add .gitignore tools docs/SOURCE_INTEGRITY.md tests/test_release_tree.py
git commit -m "chore: initialize reproducible code release"
```

---

### Task 2: Add package metadata, dependencies, and validated configurations

**Files:**

- Create: `LICENSE`
- Create: `CITATION.cff`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `configs/ideal_model.json`
- Create: `configs/physical_slm_256.json`
- Create: `configs/robustness_sweeps.json`
- Create: `configs/quick_validation.json`
- Create: `src/msf_spi/__init__.py`
- Create: `src/msf_spi/physical/config.py`
- Test: `tests/test_configuration.py`

- [ ] **Step 1: Write failing configuration tests**

Create `tests/test_configuration.py`:

```python
import json
from pathlib import Path

import pytest

from msf_spi.physical.config import PhysicalSLMConfig


ROOT = Path(__file__).resolve().parents[1]


def test_paper_configuration_matches_reported_physical_model():
    payload = json.loads((ROOT / "configs/physical_slm_256.json").read_text())
    config = PhysicalSLMConfig.from_mapping(payload)
    assert config.wavelength_m == pytest.approx(532e-9)
    assert config.slm_pixels == 256
    assert config.pixel_pitch_m == pytest.approx(8e-6)
    assert config.fill_factor == pytest.approx(0.95)
    assert config.corner_intensity == pytest.approx(0.80)
    assert config.carrier_cycles_per_fov == pytest.approx(100.0)
    assert config.first_order_iris_radius_mm == pytest.approx(1.75)
    assert config.conventional_radius == pytest.approx(32.0)
    assert config.msf_radius == pytest.approx(64.0)


def test_invalid_fill_factor_is_rejected():
    with pytest.raises(ValueError, match="fill_factor"):
        PhysicalSLMConfig(fill_factor=1.1)
```

- [ ] **Step 2: Run the tests and confirm import/configuration failures**

Run:

```powershell
python -m pytest tests/test_configuration.py -v
```

Expected: failure because the package and configuration files do not exist.

- [ ] **Step 3: Add packaging metadata**

Create `pyproject.toml` with:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "msf-spi"
version = "1.0.0"
description = "Reproducible simulations for Moire sum-frequency single-pixel imaging"
readme = "README.md"
requires-python = ">=3.11"
license = {file = "LICENSE"}
dependencies = [
  "numpy>=1.26",
  "scipy>=1.12",
  "opencv-python>=4.9",
  "matplotlib>=3.8",
  "tqdm>=4.66",
  "psutil>=5.9",
]

[project.optional-dependencies]
cuda = ["cupy-cuda12x>=13.0"]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

Create `requirements.txt` with the six core dependencies listed above and `pytest>=8.0`.

- [ ] **Step 4: Add licence and citation metadata**

Use the standard MIT licence text with `Copyright (c) 2026 MSF-SPI authors`.

Create `CITATION.cff`:

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite the associated MSF-SPI article."
title: "MSF-SPI Reproducible Simulation Code"
type: software
version: 1.0.0
date-released: 2026-09-13
authors:
  - name: "MSF-SPI authors"
license: MIT
```

- [ ] **Step 5: Create the physical configuration model and JSON files**

Implement `PhysicalSLMConfig` as an immutable dataclass based on the validated `PhysicalBandwidthConfig`, but use publication-facing field names. Include `from_mapping`, `to_mapping`, derived active width, Gaussian radius, order-lens NA, relay NA, and validation.

Write `configs/physical_slm_256.json` with the production values in the design. Write `configs/quick_validation.json` with 16 SLM pixels, oversampling 2, a 128 Fourier grid, a 3 cycles/FOV carrier, a 0.5 mm iris, conventional radius 2, MSF radius 4, and the same wavelength and fill-factor semantics.

Write `configs/ideal_model.json` with target size 512, SLM size 64, base frequency 32, maximum target frequency 64, four phases `[0, 90, 180, 270]`, circular sampling, the final global weights, 0.001-degree calibration step, and the relative angle-library path.

Write `configs/robustness_sweeps.json` using the exact final sweep arrays imported from `mode4_error_models.py`, including the three fixed random seeds.

- [ ] **Step 6: Install the package in editable mode and run tests**

Run:

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/test_configuration.py -v
```

Expected: all configuration tests pass.

- [ ] **Step 7: Commit metadata and configuration**

```powershell
git add LICENSE CITATION.cff pyproject.toml requirements.txt configs src/msf_spi tests/test_configuration.py
git commit -m "feat: add release metadata and validated configurations"
```

---

### Task 3: Port shared metrics and reconstruction primitives

**Files:**

- Create: `src/msf_spi/metrics.py`
- Create: `src/msf_spi/reconstruction/fourier.py`
- Create: `src/msf_spi/reconstruction/__init__.py`
- Test: `tests/test_reconstruction_primitives.py`

- [ ] **Step 1: Write failing primitive tests**

Create `tests/test_reconstruction_primitives.py`:

```python
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
```

- [ ] **Step 2: Run the tests and confirm missing-module failures**

Run:

```powershell
python -m pytest tests/test_reconstruction_primitives.py -v
```

Expected: failure before the modules are added.

- [ ] **Step 3: Port the validated implementations**

Copy and normalize `metrics.py`. Move `half_plane_frequency_points`, `four_step_coefficient`, centered-to-FFT phase correction, spectrum Hermitian completion, and inverse-FFT reconstruction into `reconstruction/fourier.py`. Keep all numerical formulas unchanged and replace internal experiment terminology with descriptive English names.

- [ ] **Step 4: Run tests**

```powershell
python -m pytest tests/test_reconstruction_primitives.py -v
```

Expected: three passing tests.

- [ ] **Step 5: Commit shared primitives**

```powershell
git add src/msf_spi/metrics.py src/msf_spi/reconstruction tests/test_reconstruction_primitives.py
git commit -m "feat: add shared Fourier reconstruction primitives"
```

---

### Task 4: Port and verify the pixel-resolved physical-SLM forward model

**Files:**

- Create: `src/msf_spi/physical/order_model.py`
- Create: `src/msf_spi/physical/pattern_cache.py`
- Create: `src/msf_spi/physical/frequency_mixing.py`
- Create: `src/msf_spi/physical/__init__.py`
- Test: `tests/test_physical_forward_model.py`
- Test: `tests/test_frequency_mixing.py`

- [ ] **Step 1: Write failing forward-model tests**

Create tests that assert:

```python
def test_gaussian_corner_intensity_matches_configuration():
    config = PhysicalSLMConfig.quick()
    amplitude = gaussian_amplitude(config)
    intensity = amplitude**2
    assert intensity[0, 0] / intensity.max() == pytest.approx(
        config.corner_intensity, rel=2e-2
    )


def test_fill_factor_changes_pixel_aperture_response():
    full = simulate_recovered_pattern(PhysicalSLMConfig.quick(fill_factor=1.0), 1, 0, 0)
    finite = simulate_recovered_pattern(PhysicalSLMConfig.quick(fill_factor=0.95), 1, 0, 0)
    assert not np.allclose(full.intensity, finite.intensity)


def test_physical_frequency_decomposition_is_collinear():
    result = parallel_frequency_decomposition(np.array([30.0, 40.0]), 32.0)
    assert np.cross(result["mask"], result["target"]) == pytest.approx(0.0)
    assert np.cross(result["slm"], result["target"]) == pytest.approx(0.0)
    assert result["slm"] + result["mask"] == pytest.approx(result["target"])
```

- [ ] **Step 2: Run the tests and confirm failures**

Run:

```powershell
python -m pytest tests/test_physical_forward_model.py tests/test_frequency_mixing.py -v
```

Expected: missing physical-model APIs.

- [ ] **Step 3: Port the optical forward model without altering formulas**

Port the validated phase-only encoding, pixel aperture, fill factor, Gaussian illumination, padded Fourier propagation, positive-first-order iris, finite relay pupil, and recovered-intensity functions from `physical_order_model.py`, `forward_model.py`, and `physical_msf_spi.py`. Replace implicit project-root paths with function arguments and configuration objects.

Port `PhysicalQuartetCache` using an explicit cache directory supplied by the caller. Cache keys must include all optical parameters, target vector, phase sequence, fill factor, oversampling, and FFT size.

Port `parallel_frequency_decomposition` to `physical/frequency_mixing.py` and document that it produces a zero-degree angular difference in the physical validation.

- [ ] **Step 4: Run focused tests**

```powershell
python -m pytest tests/test_physical_forward_model.py tests/test_frequency_mixing.py -v
```

Expected: all tests pass on CPU using the quick configuration.

- [ ] **Step 5: Commit the physical kernel**

```powershell
git add src/msf_spi/physical tests/test_physical_forward_model.py tests/test_frequency_mixing.py
git commit -m "feat: add pixel-resolved physical SLM forward model"
```

---

### Task 5: Port bandwidth evaluation and add the public bandwidth command

**Files:**

- Create: `src/msf_spi/physical/bandwidth.py`
- Create: `scripts/evaluate_slm_bandwidth.py`
- Test: `tests/test_bandwidth.py`
- Test: `tests/test_bandwidth_cli.py`

- [ ] **Step 1: Write failing bandwidth tests**

The unit test must evaluate a tiny configuration and assert that every result row contains frequency, angle, phase, correlation, spectral leakage, modulation-depth error, and pass/fail status. It must also assert the three production thresholds used by the final 256 x 256 code: correlation at least 0.99, leakage at most 0.01, and modulation-depth error at most 0.05. The reduced quick configuration may use 0.95, 0.10, and 0.10 solely to keep the smoke test small.

The CLI test must run:

```python
subprocess.run(
    [sys.executable, str(ROOT / "scripts/evaluate_slm_bandwidth.py"),
     "--config", str(ROOT / "configs/quick_validation.json"),
     "--output", str(tmp_path)],
    check=True,
)
assert (tmp_path / "bandwidth_summary.json").is_file()
```

- [ ] **Step 2: Confirm tests fail before the bandwidth module exists**

```powershell
python -m pytest tests/test_bandwidth.py tests/test_bandwidth_cli.py -v
```

- [ ] **Step 3: Port bandwidth search and convergence checks**

Port the final algorithms from `physical_bandwidth_256.py` and the English portions of `physical_bandwidth_256_artifacts.py`. Preserve the 36 orientations, four phase states, dual fill-factor evaluation, and higher-sampling convergence check in paper mode. Replace every Chinese artifact label and message with English.

- [ ] **Step 4: Implement the CLI**

The command accepts `--config`, `--output`, `--resume`, `--verify-only`, and `--quick`. It resolves every path from the repository root or the supplied argument, never from the caller's current directory.

- [ ] **Step 5: Run tests and quick execution**

```powershell
python -m pytest tests/test_bandwidth.py tests/test_bandwidth_cli.py -v
python scripts/evaluate_slm_bandwidth.py --config configs/quick_validation.json --output results/generated/bandwidth_quick
```

Expected: passing tests and a verified quick result directory.

- [ ] **Step 6: Commit bandwidth evaluation**

```powershell
git add src/msf_spi/physical/bandwidth.py scripts/evaluate_slm_bandwidth.py tests/test_bandwidth.py tests/test_bandwidth_cli.py
git commit -m "feat: add physical SLM bandwidth evaluation"
```

---

### Task 6: Port Eq. (10) global calibration and sparse response-matrix correction

**Files:**

- Create: `src/msf_spi/calibration/global_weights.py`
- Create: `src/msf_spi/calibration/sparse_response.py`
- Create: `src/msf_spi/calibration/__init__.py`
- Test: `tests/test_global_weights.py`
- Test: `tests/test_sparse_response.py`

- [ ] **Step 1: Write failing Eq. (10) and calibration tests**

Create a direct algebra test:

```python
def test_eq10_identity_for_nonnegative_intensity_patterns():
    a = np.linspace(0.0, 2.0 * np.pi, 31)
    b = np.linspace(0.3, 2.0 * np.pi + 0.3, 31)
    slm = (1.0 + np.cos(a)) / 2.0
    mask = (1.0 + np.cos(b)) / 2.0
    difference = (1.0 + np.cos(a - b)) / 2.0
    sum_pattern = (1.0 + np.cos(a + b)) / 2.0
    recovered = 4.0 * slm * mask - 2.0 * slm - 2.0 * mask - difference + 2.0
    assert recovered == pytest.approx(sum_pattern)
```

Add tests that recover known synthetic global weights, serialize a small sparse response matrix, reload it without numerical change, and improve reconstruction error on a deliberately coupled synthetic response.

- [ ] **Step 2: Run tests and confirm missing APIs**

```powershell
python -m pytest tests/test_global_weights.py tests/test_sparse_response.py -v
```

- [ ] **Step 3: Port global coefficient fitting**

Port `fit_global_weights` and its validation split from `physical_global_calibration.py`. Define the five measurable channels explicitly as Moire total, SLM fundamental, mask fundamental, difference frequency, and uniform illumination. Keep coefficient order and normalization in one exported constant to prevent silent rearrangement.

- [ ] **Step 4: Port sparse response calibration**

Port `FrozenResponse`, kernel-response extraction, sparse matrix construction, regularized solver, serialization, fingerprints, and diagnostics from `physical_response_matrix.py`, `mode4_sparse_response.py`, and the validated physical pipeline. Public names must describe response calibration rather than internal mode numbers.

- [ ] **Step 5: Run calibration tests**

```powershell
python -m pytest tests/test_global_weights.py tests/test_sparse_response.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit calibration modules**

```powershell
git add src/msf_spi/calibration tests/test_global_weights.py tests/test_sparse_response.py
git commit -m "feat: add global and sparse response calibration"
```

---

### Task 7: Port the physical SPI/MSF-SPI comparison and reproduce Figure R5

**Files:**

- Create: `src/msf_spi/physical/comparison.py`
- Create: `scripts/reproduce_physical_comparison.py`
- Test: `tests/test_physical_comparison.py`
- Test: `tests/test_physical_comparison_cli.py`

- [ ] **Step 1: Write failing comparison tests**

Use the quick configuration and a generated square-and-bars target. Assert that the returned result contains the original, ideal MSF-band-limited reference, conventional SPI reconstruction, MSF-SPI reconstruction, metrics relative to the original, metrics relative to the matched band-limited reference, and the exact sampled point counts.

The CLI test must assert creation of `comparison.png`, `metrics.json`, `configuration.json`, and `run_manifest.json`.

- [ ] **Step 2: Run tests and confirm failures**

```powershell
python -m pytest tests/test_physical_comparison.py tests/test_physical_comparison_cli.py -v
```

- [ ] **Step 3: Port the final comparison pipeline**

Port the validated path from `physical_fmax_msf_experiment.py` and the final response-matrix integration. Preserve circular half-plane sampling, Hermitian completion, global weights, sparse response correction, and matched-band metrics. Do not introduce reference-image angle optimization into this physical path.

- [ ] **Step 4: Implement the reproduction command**

The CLI accepts `--config`, `--image`, `--response-matrix`, `--output`, `--quick`, `--rebuild-response`, `--seed`, and `--verify-only`. By default it loads the supplied physical response matrix; `--rebuild-response` regenerates it from known calibration patterns.

- [ ] **Step 5: Run tests and the quick comparison**

```powershell
python -m pytest tests/test_physical_comparison.py tests/test_physical_comparison_cli.py -v
python scripts/reproduce_physical_comparison.py --quick --config configs/quick_validation.json --image data/targets/USAF-1951.jpg --output results/generated/physical_comparison_quick
```

- [ ] **Step 6: Commit the physical comparison**

```powershell
git add src/msf_spi/physical/comparison.py scripts/reproduce_physical_comparison.py tests/test_physical_comparison.py tests/test_physical_comparison_cli.py
git commit -m "feat: add physical SPI and MSF-SPI comparison"
```

---

### Task 8: Port the final six-error robustness pipeline and reproduce Figure R6

**Files:**

- Create: `src/msf_spi/robustness/cases.py`
- Create: `src/msf_spi/robustness/pipeline.py`
- Create: `src/msf_spi/robustness/artifacts.py`
- Create: `src/msf_spi/robustness/__init__.py`
- Create: `scripts/reproduce_robustness.py`
- Test: `tests/test_robustness_cases.py`
- Test: `tests/test_robustness_pipeline.py`
- Test: `tests/test_robustness_cli.py`

- [ ] **Step 1: Write failing robustness tests**

Assert that the configured families are exactly:

```python
{
    "mask_angle",
    "mask_frequency",
    "slm_static_phase",
    "detector_noise",
    "calibration_mismatch",
    "conjugate_defocus",
}
```

Assert that identical seeds reproduce identical phase maps, angle errors, coefficient mismatches, and detector-noise sequences. Assert that measurement perturbations do not modify nominal reconstruction patterns or the frozen response matrix.

The CLI smoke test runs one reduced level per family and checks `robustness_summary.png`, `summary_metrics.json`, `error_realizations.json`, and `verification_report.json`.

- [ ] **Step 2: Run tests and confirm failures**

```powershell
python -m pytest tests/test_robustness_cases.py tests/test_robustness_pipeline.py tests/test_robustness_cli.py -v
```

- [ ] **Step 3: Port cases and measurement-side perturbations**

Port the final level arrays and seeds. Preserve the final semantics:

- one uniformly distributed positioning error per commanded mask orientation, shared by its four phase steps;
- systematic mask-frequency error;
- zero-mean random static SLM phase map fixed across acquisition;
- additive Gaussian bucket noise specified by AC-SNR;
- random multiplicative global-weight mismatch; and
- angular-spectrum propagation for conjugate-plane defocus.

- [ ] **Step 4: Port the corrected final reconstruction path**

Use the same frozen sparse matrix calibrated for the nominal physical model for every perturbation. Document and enforce that the robustness curves represent the final algorithm after sparse response-matrix correction, not the earlier uncorrected reconstruction.

- [ ] **Step 5: Port English artifact generation and CLI**

Translate every Chinese plot label, log message, and supplementary draft fragment in the three artifact writers. The public command accepts `--config`, `--sweeps`, `--image`, `--response-matrix`, `--output`, `--quick`, `--resume`, `--seed`, and `--verify-only`.

- [ ] **Step 6: Run tests and quick robustness execution**

```powershell
python -m pytest tests/test_robustness_cases.py tests/test_robustness_pipeline.py tests/test_robustness_cli.py -v
python scripts/reproduce_robustness.py --quick --config configs/quick_validation.json --sweeps configs/robustness_sweeps.json --image data/targets/USAF-1951.jpg --output results/generated/robustness_quick
```

- [ ] **Step 7: Commit robustness pipeline**

```powershell
git add src/msf_spi/robustness scripts/reproduce_robustness.py tests/test_robustness_cases.py tests/test_robustness_pipeline.py tests/test_robustness_cli.py
git commit -m "feat: add calibrated six-error robustness analysis"
```

---

### Task 9: Port the final idealized reconstruction without private paths or Chinese text

**Files:**

- Create: `src/msf_spi/ideal/full_simulation.py`
- Create: `src/msf_spi/ideal/patterns.py`
- Create: `src/msf_spi/ideal/response.py`
- Create: `src/msf_spi/ideal/dual_angle.py`
- Create: `src/msf_spi/ideal/__init__.py`
- Create: `scripts/reproduce_ideal_reconstruction.py`
- Test: `tests/test_ideal_model.py`
- Test: `tests/test_ideal_cli.py`

- [ ] **Step 1: Write failing ideal-model tests**

Tests must assert that the ideal configuration uses a 64 x 64 programmable grid, 512 x 512 output grid, circular frequency sampling, base radius 32, extended radius 64, and the supplied reference-image angle library. A reduced synthetic run must create four phase-shifted nonnegative patterns per frequency and must improve the sparse-system forward residual relative to the uncorrected reconstruction.

- [ ] **Step 2: Run tests and confirm failures**

```powershell
python -m pytest tests/test_ideal_model.py tests/test_ideal_cli.py -v
```

- [ ] **Step 3: Create a functionally frozen English source copy**

Copy `Moire_SPI_260603_sparse_response.py` into `ideal/full_simulation.py`. Change imports to package-relative imports. Translate all Chinese comments, docstrings, printed messages, plot titles, menu text, and configuration explanations to English without modifying numerical expressions, branch conditions, coefficient order, array shapes, or sampling rules.

Rename public entry functions so that “Mode 4” becomes `run_ideal_msf_spi` and “Mode 10” becomes `run_noise_sensitivity`. Compatibility aliases may remain private only where needed to verify numerical equivalence.

- [ ] **Step 4: Port ideal helper modules**

Port the relevant functions from `mode4_gpu.py`, `mode4_sparse_response.py`, and `ideal_mode4_dual_angle.py` into descriptive modules. Replace the dynamic import of the dated working script with a normal package import. Replace parent-directory assumptions with explicit paths from configuration.

- [ ] **Step 5: Add the ideal reproduction CLI**

The CLI accepts `--config`, `--image`, `--angle-library`, `--output`, `--angle-step`, `--response-matrix`, `--rebuild-response`, `--quick`, `--seed`, and `--verify-only`. User-facing output must call this the idealized algebraic MSF-SPI model and state that its angle library is reference-image based.

- [ ] **Step 6: Verify numerical equivalence on a reduced fixed input**

Run the original and released functions on the same small target and compare the four intensity components, extracted complex coefficients, and final normalized reconstruction with relative tolerance `1e-7` and absolute tolerance `1e-8`.

- [ ] **Step 7: Run ideal tests**

```powershell
python -m pytest tests/test_ideal_model.py tests/test_ideal_cli.py -v
```

Expected: all tests pass and no private source import occurs.

- [ ] **Step 8: Commit the ideal model**

```powershell
git add src/msf_spi/ideal scripts/reproduce_ideal_reconstruction.py tests/test_ideal_model.py tests/test_ideal_cli.py
git commit -m "feat: add English idealized MSF-SPI reproduction"
```

---

### Task 10: Add exact inputs, precomputed calibration, reference outputs, and checksums

**Files:**

- Create: `data/README.md`
- Create: `data/targets/USAF-1951.jpg`
- Create: `data/angle_library/optimal_angle_differences.json`
- Create: `data/precomputed/physical_256_response_matrix.npz`
- Create: `data/precomputed/global_weights.json`
- Create: `data/precomputed/checksums.sha256`
- Create: `results/reference/physical_comparison/*`
- Create: `results/reference/robustness/*`
- Create: `docs/RESULTS_INDEX.md`
- Test: `tests/test_reference_artifacts.py`

- [ ] **Step 1: Write failing artifact tests**

Create tests that load the USAF image, angle library, global weights, and sparse matrix; verify nonempty dimensions and finite values; verify every SHA-256 entry; and assert that the reference metrics contain the reported physical baseline fields.

- [ ] **Step 2: Run tests and confirm missing-artifact failures**

```powershell
python -m pytest tests/test_reference_artifacts.py -v
```

- [ ] **Step 3: Copy only the approved compact artifacts**

Use `Copy-Item -LiteralPath` for the exact files recorded in `docs/source_manifest.json`. Do not copy shared-measurement NPZ files, static-phase caches, pipeline caches, per-case image folders, historical result trees, Cameraman, or Lena.

Rename copied files to the English names in the file map. Preserve binary bytes exactly and record original and released hashes.

- [ ] **Step 4: Write data and results documentation**

`data/README.md` must identify the role and provenance of each file, state that the USAF target was the exact simulation input, and explicitly instruct the authors to confirm its redistribution rights before public upload. `RESULTS_INDEX.md` must link each reference figure and metrics file to the command that regenerates it.

- [ ] **Step 5: Generate and verify checksums**

Generate standard SHA-256 lines using relative POSIX-style paths. Run the artifact test and expect all checks to pass.

- [ ] **Step 6: Commit data and compact results**

```powershell
git add data results/reference docs/RESULTS_INDEX.md tests/test_reference_artifacts.py
git commit -m "data: add validated inputs and reference artifacts"
```

---

### Task 11: Write publication-facing documentation and quick-start orchestration

**Files:**

- Create: `README.md`
- Create: `docs/MODEL_ASSUMPTIONS.md`
- Create: `docs/CALIBRATION.md`
- Create: `docs/REPRODUCIBILITY.md`
- Create: `docs/FILE_PROVENANCE.md`
- Create: `scripts/run_quick_validation.py`
- Test: `tests/test_documentation.py`
- Test: `tests/test_quick_validation.py`

- [ ] **Step 1: Write failing documentation audits**

Create `tests/test_documentation.py` that reads public `.py`, `.md`, `.json`, `.toml`, `.yaml`, `.yml`, `.cff`, and `.txt` files, excluding Git internals and the source-integrity manifest. Assert that no Han characters occur and that neither private source root appears in executable source or user-facing documentation. Permit private roots only in `docs/source_manifest.json`, `docs/SOURCE_INTEGRITY.md`, and `docs/FILE_PROVENANCE.md` because those files document traceability.

Create `tests/test_quick_validation.py` that runs `scripts/run_quick_validation.py --output <tmp>` and verifies a final `validation_report.json` with `status` equal to `passed`.

- [ ] **Step 2: Run audits and confirm they fail before documentation exists**

```powershell
python -m pytest tests/test_documentation.py tests/test_quick_validation.py -v
```

- [ ] **Step 3: Write the README**

Cover these exact topics:

1. scientific purpose and cautious simulation-only status;
2. repository map;
3. Python installation and optional CUDA installation;
4. five-minute quick validation;
5. idealized model reproduction;
6. physical SLM bandwidth evaluation;
7. physical SPI/MSF-SPI comparison;
8. global and sparse response calibration;
9. six-error robustness reproduction;
10. expected full-run memory and time considerations;
11. reference results and numerical tolerances;
12. data licence warning;
13. citation and MIT software licence.

- [ ] **Step 4: Write the model and calibration documents**

`MODEL_ASSUMPTIONS.md` must contain a comparison table between the idealized and physical models. It must state that the physical model uses zero angular difference and does not use Eq. (8) optimization because positive-first-order filtering yields nearly sinusoidal illumination.

`CALIBRATION.md` must separately explain global Eq. (10) weights, reference-image angle optimization, and sparse system-response calibration, including simulation and experimental acquisition requirements.

`REPRODUCIBILITY.md` must map every reviewer-response figure and metric to one command and configuration.

`FILE_PROVENANCE.md` must map every released source module to its original working file and describe whether it was copied, translated, split, or renamed.

- [ ] **Step 5: Implement quick orchestration**

`run_quick_validation.py` must run bandwidth, physical comparison, robustness, and an ideal-model smoke case in separate output folders; collect durations and pass/fail checks; and write one JSON report. It must stop on the first failure and return a nonzero exit code.

- [ ] **Step 6: Run documentation and quick validation tests**

```powershell
python -m pytest tests/test_documentation.py tests/test_quick_validation.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit documentation**

```powershell
git add README.md docs scripts/run_quick_validation.py tests/test_documentation.py tests/test_quick_validation.py
git commit -m "docs: add complete reproduction guide"
```

---

### Task 12: Add final release verification and prove source directories were untouched

**Files:**

- Create: `scripts/verify_release.py`
- Modify: `tools/build_source_manifest.ps1`
- Test: `tests/test_verify_release.py`

- [ ] **Step 1: Write a failing release-verifier test**

The test invokes `verify_release.py --quick --report <tmp>/report.json` and asserts checks for package import, unit tests, English-only content, private-path isolation, data checksums, reference-artifact loading, quick execution, and original-source hash stability.

- [ ] **Step 2: Run the test and confirm the verifier is missing**

```powershell
python -m pytest tests/test_verify_release.py -v
```

- [ ] **Step 3: Implement the verifier**

The verifier must run each check as a subprocess, capture command, exit code, duration, and a bounded output tail, then write a machine-readable JSON report. It must not modify or normalize the original source files. Source-integrity checking must rehash the exact manifest entries and fail on any difference.

- [ ] **Step 4: Run the complete unit-test suite**

```powershell
python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 5: Run verification from outside the repository**

```powershell
Set-Location $env:TEMP
python scripts/verify_release.py --quick --report results/generated/release_verification.json
```

Expected: a zero exit code and report status `passed`, proving that commands do not depend on the repository being the current directory.

- [ ] **Step 6: Scan the release manually for prohibited content**

Run:

```powershell
rg -n "[\p{Han}]" . -g "!docs/source_manifest.json" -g "!docs/SOURCE_INTEGRITY.md" -g "!docs/FILE_PROVENANCE.md" -g "!.git/**"
rg -n "Moire_SPI_260603|Mode.?4|Mode.?10|最终综合测试|审稿意见" src scripts README.md configs tests
```

Expected: no matches.

- [ ] **Step 7: Review repository size and exclude accidental caches**

Run:

```powershell
Get-ChildItem -Recurse -File | Sort-Object Length -Descending | Select-Object -First 30 FullName,Length
git status --short
```

Expected: only the approved sparse matrix and compact reference images are multi-megabyte files; no generated cache is staged.

- [ ] **Step 8: Commit final verification**

```powershell
git add scripts/verify_release.py tools/build_source_manifest.ps1 tests/test_verify_release.py results/generated/release_verification.json
git commit -m "test: verify reproducible release end to end"
```

- [ ] **Step 9: Record the final repository state**

Run:

```powershell
git status --short
git log --oneline --decorate -12
```

Expected: clean working tree and a linear sequence of focused release commits.

---

## Final Self-Review Checklist

- Every design requirement is covered by Tasks 1-12.
- The ideal and physical models remain separate in source, configuration, commands, and documentation.
- The 256 x 256 physical model parameters match the validated final simulation.
- The physical model explicitly uses a zero-degree angular difference and omits Eq. (8) optimization.
- The robustness pipeline explicitly uses the final sparse-response-corrected reconstruction.
- Original files are read and hashed but never edited.
- Public paths do not depend on Chinese folder names.
- Large transient caches are excluded.
- The exact target, angle library, sparse matrix, compact metrics, and reference figures are retained.
- CPU quick validation is mandatory; CUDA is optional.
- Public text and code are English-only.
- Data redistribution uncertainty is disclosed before public upload.
