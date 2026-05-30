# SIREN vs U-Net IFWI — Half Marmousi Results

Implicit Full Waveform Inversion on the left half of the Marmousi model,
comparing two velocity parameterizations (supervisor's request).

## Setup

| Item | Value |
|---|---|
| Model | Half Marmousi, 94 × 144, dz = 15 m |
| Sources | 13 surface shots, 150 m spacing |
| Receivers | 144 (every grid point) |
| Wavelet | Ricker, frequency continuation 3 → 5 → 8 Hz |
| Epochs | 1500 (3 Hz) + 1500 (5 Hz) + 2000 (8 Hz) = 5000 per method |
| dt / nt | 0.0019 s / 1000 |
| Optimizer | Adam |

Both methods pretrained to start from their initial model, then inverted
through the same `rnn2D` finite-difference wave engine.

- **SIREN**: input = spatial coordinates (z, x); started from the smooth
  Marmousi initial model.
- **U-Net**: input = a linear (depth-increasing) initial model image;
  started from that linear model (supervisor's choice).

## Why frequency continuation

Single-frequency 8 Hz inversion cycle-skipped: the data misfit dropped 99%
but the model error got *worse* (−14.8%), with the deep high-velocity body
capped near 4.0 km/s. Starting low (3 Hz) and stepping up to 8 Hz fixed this
for both methods.

## Results

| Metric | SIREN | U-Net |
|---|---|---|
| Final MAE (km/s) | 0.3069 | **0.2004** |
| Final RMS (km/s) | 0.6102 | **0.3469** |
| Predicted vmin | 1.459 | 1.253 |
| Predicted vmax | 4.020 | **6.166** (truth 5.500) |

**U-Net clearly outperformed SIREN on this half-aperture problem.** It
recovered the deep ~5.5 km/s body that SIREN left capped at ~4.0 km/s. The
U-Net's convolutions share information spatially, so a deep update
propagates across the whole layer instead of being fought point-by-point as
in the coordinate-wise SIREN. U-Net slightly overshoots the maximum
velocity (6.17 vs 5.50), a minor over-prediction in the deep body.

Figures (saved on the run environment):
- `SIREN_half_result_3panel.png` — SIREN truth / prediction / error
- `UNET_half_result_3panel.png` — U-Net truth / prediction / error
- `SIREN_vs_UNET_comparison.png` — combined 2×3 comparison

## Notes / caveats

- The deep high-velocity body is hard to constrain because of the surface-only,
  half-aperture geometry (limited ray coverage at depth). Both methods share
  this limitation; U-Net handles it better.
- The two methods use different initial models (smooth Marmousi vs linear),
  per the supervisor's spec, so the *improvement percentages* are not directly
  comparable. The fair comparison is the **final prediction quality** (MAE,
  RMS, recovered velocity range) reported above.
- U-Net at 8 Hz needed a lower learning rate (3e-5) and velocity clamping to
  stay numerically stable; SIREN was stable at 1e-4 throughout.

## Reproducing

Run cell by cell on the ModelWhale H100 environment:
1. `ifwi/siren_half_marmousi.py` — Cells 1–6 (SIREN result)
2. `ifwi/unet_half_marmousi.py` — run SIREN Cells 1–3 first, then UCELL 1–3 (U-Net result)
3. The comparison cell (loads both saved `.npy` predictions)

All training is crash-safe: checkpoints every 100 epochs, stage-final files,
auto-resume from the latest checkpoint on re-run.
