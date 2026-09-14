# Data files

targets/USAF-1951.jpg is the exact grayscale resolution target used in the validated simulations. It is included as a publicly available test resource to support reproducibility.

`angle_library/optimal_angle_differences.json` contains 51,408 phase-resolved entries. It was obtained by one-time reference-image-based optimization and is used only by the idealized algebraic model. It is not a strictly object-independent system calibration.

`precomputed/physical_256_global_weights.json` stores the five global Eq. (10) response weights fitted for the final physical model. `physical_256_response_matrix.npz` stores the object-independent sparse system-response matrix used in the final calibrated physical reconstruction and robustness sweep.

The three `ideal_response_matrix_0p001deg*` files store the sparse matrix, per-frequency responses, and metadata for the 0.001-degree idealized angular condition. They permit inspection of the calibration used for the archived angular-quantization comparison without distributing the tens of gigabytes of temporary pattern batches.

`precomputed/checksums.sha256` covers all distributed inputs, calibrations, and compact reference outputs. Paths are relative to the repository root.
