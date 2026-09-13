# File provenance

The private working directories were treated as read-only. `source_manifest.json` records their absolute paths, byte counts, and SHA-256 hashes. Public modules are cleaned scientific interfaces rather than a wholesale copy of dated exploratory scripts.

| Released component | Source role | Transformation |
|---|---|---|
| `src/msf_spi/ideal/patterns.py` | Optimized five-term illumination generator | Extracted, renamed, documented, and made path-independent; numerical coefficient order retained |
| `src/msf_spi/ideal/simulation.py` | Full circular-frequency acquisition and reconstruction | Split from the exploratory driver into a noninteractive API |
| `src/msf_spi/physical/order_model.py` | Phase-only SLM propagation | Imports normalized and configuration made explicit |
| `src/msf_spi/physical/bandwidth.py` | Physical bandwidth search | Artifact generation separated from the numerical kernel |
| `src/msf_spi/physical/comparison.py` | SPI/MSF-SPI comparison | Converted to a reusable function and connected to optional precomputed `H` |
| `src/msf_spi/calibration/global_weights.py` | Five-term Eq. (10) regression | Consolidated into one English API |
| `src/msf_spi/calibration/sparse_response.py` | Per-frequency and sparse response calibration | Consolidated and renamed without changing the linear algebra |
| `src/msf_spi/robustness/*` | Six perturbation families and final sweep | Split into error definitions, pipeline, and artifacts; frozen-`H` semantics retained |

Temporary mask batches, measurement caches, historical outputs, unrelated test images, and intermediate scripts were intentionally excluded. They are large, redundant, or not required to reproduce the final algorithms. The precomputed matrices and compact summaries retain the expensive calibrated state needed for inspection and reuse.
