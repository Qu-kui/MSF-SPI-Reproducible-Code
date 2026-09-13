# Model assumptions

| Property | Idealized algebraic model | Pixel-resolved wave-optics model |
|---|---|---|
| Primary purpose | Test frequency mixing, angle-library transfer, and reconstruction | Test attainable SLM illumination and bandwidth under representative optical constraints |
| Programmable grid | 64 x 64 | 256 x 256 in the production configuration |
| Sample-plane grid | 512 x 512 | 1024 x 1024 |
| SLM pixel representation | Exact nearest-neighbor pixel replication; no bicubic interpolation | Finite 8-um pixel pitch and 0.95 fill factor |
| Incident beam | Implicitly uniform | Gaussian amplitude with corner intensity equal to 80% of peak intensity |
| Phase encoding | Not propagated explicitly | Phase-only hologram with a blazed carrier |
| Diffraction orders | Positive first order is selected mathematically | Positive first order is selected by a finite circular Fourier-plane iris |
| Relay optics | Infinite effective numerical aperture | Finite 25.4-mm clear apertures and 100-mm focal-length relay lenses |
| Frequency-vector geometry | Reference-image-optimized angular difference | Collinear decomposition; angular difference is 0 degrees |
| Defocus | Absent | Angular-spectrum propagation in the robustness model |
| Calibration | Global Eq. (10), per-frequency response, sparse matrix | Global Eq. (10), per-frequency response, sparse matrix |

Both models use scalar, coherent, monochromatic propagation and a linear bucket detector. Polarization dependence, wavelength bandwidth, surface roughness, SLM temporal response, stage settling, detector nonlinearity, and laboratory drift are not fully modeled.

The idealized model assumes that the desired positive diffraction order can be isolated completely and that the relay transmits the represented bandwidth. Its block structure is therefore a numerical representation of independently addressable SLM pixels, not a prediction of the optical intensity obtained after a finite pupil.

The physical model uses a 1:1 relay between conjugate planes. The SLM-to-sample and sample-to-mask transverse magnifications are unity in magnitude, with coordinate inversion included consistently. Because the recovered positive-order illumination is already approximately sinusoidal, the reference-image angular-difference optimization is not applied in this model. This is a modeling choice, not proof that no angular calibration will ever be needed experimentally.
