# Source Integrity

The release is assembled in a separate repository. The two working directories used to produce the manuscript and reviewer-response simulations are treated as read-only inputs.

`tools/build_source_manifest.ps1` records the absolute source path, intended release path, byte count, and SHA-256 digest of every selected input. Before rebuilding this private provenance manifest, set `MSF_SPI_REVIEW_SOURCE_ROOT` and `MSF_SPI_WORK_SOURCE_ROOT` to the two local source directories. The manifest is generated before packaging and checked again during final verification. A hash mismatch is treated as a release failure.

Absolute working paths appear only in the private provenance records. Executable release code, configurations, and user-facing instructions do not depend on those paths.
