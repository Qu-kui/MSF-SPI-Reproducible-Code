# MSF-SPI Reproducible Code Release Design

## Purpose

Create a clean, English-only research-code package that reproduces the final idealized MSF-SPI studies, the pixel-resolved physical-SLM validation, the coefficient and sparse-response calibrations, and the six robustness analyses used in the manuscript revision and reviewer responses.

The release is assembled from the working files under:

- `D:/DeskTop/审稿意见`
- `D:/DeskTop/我的工作/摩尔纹基底单像素成像-done/最终综合测试`

The source directories remain unchanged. All publication edits are made only inside `D:/DeskTop/审稿意见/MSF-SPI-Reproducible-Code`.

## Intended Audience

The primary audience is a reviewer or researcher who wants to:

1. understand the difference between the idealized and physically constrained models;
2. reproduce the principal reconstruction comparisons and robustness plots;
3. inspect the calibration and sparse response-matrix correction;
4. run a small verification case without waiting for the full paper-scale computation; and
5. identify the assumptions that still require experimental validation.

## Scope

### Included

- The final idealized MSF-SPI reconstruction used to study the algebraic mixing model.
- The reference-image angle-difference library used by the idealized model.
- The 256 x 256 pixel-resolved phase-only SLM forward model.
- Gaussian incident illumination with 80% corner intensity over the active SLM region.
- An 8 micrometre SLM pixel pitch and 0.95 fill factor.
- Blazed-carrier encoding, scalar propagation, finite-aperture filtering, and selection of the positive first diffraction order.
- The physically constrained spatial-frequency bandwidth evaluation.
- Conventional SPI and MSF-SPI reconstruction comparisons.
- Global Eq. (10) coefficient calibration.
- Object-independent sparse system-response matrix calibration and regularized reconstruction.
- The six final robustness sweeps: mask angle, mask frequency, SLM static phase, detector noise, calibration mismatch, and conjugate-plane defocus.
- Reproduction scripts for the final reference figures and metrics.
- Unit tests and small smoke tests.

### Excluded

- Historical dated script versions that were superseded by the final pipelines.
- Debugging images, temporary document extraction folders, raw intermediate masks, and multi-gigabyte measurement caches.
- Alternative three-mask experiments not used in the submitted revision.
- Cameraman and Lena test images unless their redistribution licence is independently verified.
- Blender scene construction code, because it documents the conceptual optical layout rather than the numerical reconstruction pipeline. The final rendered optical-layout figure may be referenced separately in the supplementary material.

## Scientific Model Separation

The package must keep two models visibly separate.

### Idealized algebraic model

This model assumes exact numerical access to the desired illumination components and uses the reference-image-based angular-difference library associated with Eq. (8). It represents the assumptions used in the original numerical proof of principle. The public interface will use the term `ideal_model`; internal labels such as “Mode 4” will not appear in user-facing commands or documentation.

The ideal-model angle library is not described as an object-independent system calibration. Its provenance and dependence on the prespecified reference image are stated explicitly.

### Pixel-resolved physical-SLM model

This model obtains the sample-plane illumination by phase-only encoding and scalar wave propagation instead of by interpolating an ideal intensity image. The production configuration uses:

- wavelength: 532 nm;
- SLM array: 256 x 256 pixels;
- pixel pitch: 8 micrometres;
- fill factor: 0.95;
- Gaussian illumination corner intensity: 0.80;
- numerical oversampling: 4 samples per SLM pixel;
- Fourier grid: 2048 x 2048;
- first relay: two 200 mm focal-length lenses;
- second relay: two 100 mm focal-length lenses;
- lens clear diameter: 25.4 mm;
- blazed carrier: 100 cycles per field of view;
- positive-first-order iris radius: 1.75 mm;
- conventional SPI radius: 32 cycles per field of view;
- MSF-SPI radius: 64 cycles per field of view.

The physical validation uses collinear frequency-vector decomposition. The external-mask vector is parallel to the target vector, so the angular difference is zero degrees. Because positive-first-order filtering suppresses the SLM pixel-grid diffraction components and produces an approximately sinusoidal sample-plane illumination, the reference-image angle optimization of Eq. (8) is not applied to this physical validation.

The two models are complementary and must not be presented as numerically identical implementations.

## Repository Structure

```text
MSF-SPI-Reproducible-Code/
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- pyproject.toml
|-- requirements.txt
|-- .gitignore
|-- configs/
|   |-- ideal_model.json
|   |-- physical_slm_256.json
|   `-- robustness_sweeps.json
|-- src/msf_spi/
|   |-- __init__.py
|   |-- metrics.py
|   |-- ideal/
|   |-- physical/
|   |-- calibration/
|   |-- reconstruction/
|   `-- robustness/
|-- scripts/
|   |-- reproduce_ideal_reconstruction.py
|   |-- evaluate_slm_bandwidth.py
|   |-- reproduce_physical_comparison.py
|   |-- reproduce_robustness.py
|   `-- run_quick_validation.py
|-- data/
|   |-- README.md
|   |-- targets/
|   |-- angle_library/
|   `-- precomputed/
|-- results/reference/
|-- docs/
|   |-- MODEL_ASSUMPTIONS.md
|   |-- CALIBRATION.md
|   |-- REPRODUCIBILITY.md
|   |-- FILE_PROVENANCE.md
|   `-- superpowers/
`-- tests/
```

## Source Selection and Renaming

The final package will be derived primarily from the following working files:

- `Moire_SPI_260603_sparse_response.py` for the final idealized reconstruction and sparse-response integration;
- `optimize_coefficients.py` for the original coefficient-fitting procedure;
- `Optimal_angle_diffs_SPI_circle_radial_stripes_optimized.json` for the ideal-model angular-difference library;
- `MSF_SPI_waveoptics_robustness/physical_bandwidth_config.py`;
- `physical_order_model.py`;
- `physical_pattern_cache.py`;
- `physical_bandwidth_256.py` and its artifact writer;
- `physical_msf_spi.py`;
- `physical_fmax_msf_experiment.py`;
- `physical_global_calibration.py`;
- `physical_response_matrix.py`;
- `physical_robustness_256.py`;
- `physical_robustness_256_pipeline.py` and its artifact writer;
- `metrics.py`; and
- the corresponding focused tests.

Dated filenames, “Mode 4”, “Mode 10”, and “legacy” terminology will be replaced in the public interface by scientific names. Functionality will be preserved while imports, paths, comments, docstrings, plot labels, and console messages are normalized to English.

## Data and Precomputed Artifacts

The package will include compact inputs and reference outputs required for direct verification:

- the exact USAF-1951 target used by the final simulations, subject to a redistribution notice in `data/README.md`;
- the final angle-difference JSON for the idealized model;
- global calibration coefficients;
- the approximately 6 MB sparse response matrix for the 256 x 256 physical model;
- configuration and metric JSON files;
- the final physical comparison and robustness figures.

Large shared-measurement arrays, phase-realization caches, and regenerable pipeline caches will be excluded. Each included binary artifact will have a SHA-256 checksum and a script capable of regenerating it.

## Commands and Reproducibility Levels

Every public script will accept explicit input, output, configuration, random-seed, and resume arguments. No script may rely on the original Chinese directory names or on the current working directory.

Two execution levels will be documented:

- `quick`: reduced numerical grids and fewer sweep points for installation and smoke testing;
- `paper`: the full parameters used for the reported figures and metrics.

The README will state that the full physical and robustness runs are computationally expensive. CPU execution is the reference path; CUDA acceleration is optional and must fall back cleanly when CuPy is unavailable.

## Calibration Documentation

The calibration documentation will distinguish three operations:

1. Global Eq. (10) weighting coefficients compensate for relative gains among measurable intensity classes.
2. The ideal-model angular-difference library is reference-image-based parameter optimization.
3. The sparse response matrix maps the known illumination-response basis to measured Fourier coefficients and is independent of the unknown reconstruction target once characterized for a fixed system.

The documentation will describe how each calibration is generated in simulation, what would be measured experimentally, when recalibration is required, and which precomputed files are supplied.

## Error Handling

Entry-point scripts will fail with concise messages when an input image, angle library, precomputed response matrix, or required dependency is missing. Configuration validation will reject invalid SLM dimensions, fill factors, apertures, frequency radii, or robustness levels before expensive computation begins. Output directories will be created without overwriting source data; an explicit resume flag will control reuse of compatible cached artifacts.

## Testing and Verification

Testing will cover:

- configuration validation and physical-unit conversions;
- frequency-point enumeration and collinear decomposition;
- four-step demodulation and Eq. (10) algebra;
- positive-first-order propagation on a small grid;
- global coefficient fitting;
- sparse response-matrix construction, serialization, and reconstruction;
- deterministic robustness realizations under a fixed seed;
- command-line smoke tests in quick mode;
- absence of Chinese characters in public text and source files;
- absence of absolute paths referring to the two private working directories; and
- checksum verification for included precomputed artifacts.

The final acceptance check will run the unit-test suite, execute quick validation from a directory outside the repository, and compare generated metrics against stored quick-reference tolerances.

## Documentation and Licensing

The package will use an MIT licence unless replaced by the authors before publication. `CITATION.cff` will contain neutral placeholder-free project metadata that can be updated with the final paper DOI. Because no Git repository currently exists in either source directory, the release folder will be initialized as a separate Git repository only after its contents pass the verification checks.

`FILE_PROVENANCE.md` will map every released module and data artifact to its working source, documenting whether it was copied, renamed, translated, split, or regenerated. This preserves traceability without exposing obsolete files.

## Acceptance Criteria

The release is complete when:

1. all public filenames, comments, documentation, messages, and figure labels are English;
2. no public code contains an absolute private path;
3. the ideal and physical models can be run through descriptive commands;
4. quick validation passes on CPU without pre-existing caches;
5. the supplied physical sparse matrix and reference outputs pass checksum checks;
6. the selected unit tests pass;
7. the README explains how to reproduce the ideal reconstruction, bandwidth evaluation, physical comparison, and six robustness sweeps; and
8. the original source directories remain byte-for-byte untouched by the packaging process.
