# Calibration procedures

Three distinct calibrations appear in the study. They should not be described as one five-step calibration.

## Global five-term weights

Equation (10) combines five measurable nonnegative illumination classes: total Moire illumination, SLM-only illumination, mask-only illumination, difference-frequency illumination, and uniform zero-frequency illumination. `global_weights.py` fits one set of five system-level weights from known calibration patterns by robust linear regression. The released 256 x 256 weights are stored in `data/precomputed/physical_256_global_weights.json`.

These weights are system-specific rather than object-specific. In an experiment they can remain fixed while wavelength, illumination profile, SLM response, mask, relay geometry, alignment, and detector gain remain stable. Recalibration is required after appreciable system drift.

## Reference-image angular optimization

The idealized model uses `data/angle_library/optimal_angle_differences.json`. This library was optimized once with a prespecified reference image containing multiple orientations and spatial frequencies, then fixed for all later test objects. Its transferability depends on the selected reference image. It must therefore be called reference-image-based parameter optimization, not strictly object-independent system calibration.

The pixel-resolved physical model does not use this library. Its SLM and external-mask frequency vectors are collinear, so the angular difference is 0 degrees.

## Sparse system-response matrix

Let the vector of desired complex Fourier coefficients be represented by interleaved real and imaginary components. The calibrated linear system is `y = H x`, where each row pair is obtained from the measured or simulated four-phase illumination kernel for one target frequency. The diagonal and nearby off-diagonal terms describe DC leakage, desired gain, conjugate gain, and cross-frequency coupling. Reconstruction solves a regularized sparse inverse problem using one frozen matrix `H`.

The unknown object is not used to construct `H`. In simulation, `H` is computed directly from the illumination fields at the sample plane. In an experiment, equivalent information can be obtained by measuring those patterns with a calibrated camera temporarily placed at the sample plane or by scanning known spatial basis targets and recording the bucket response. The calibration must cover the same ordered frequency basis, phase convention, optical configuration, and normalization used during acquisition.

The six-factor robustness study deliberately perturbs the simulated measurement while retaining the nominal global weights and the same sparse `H`. It therefore measures calibration transfer under mismatch rather than granting each error condition a new fit.
