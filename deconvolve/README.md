# deconvolve

Deconvolution of the average physics waveform by the SPE template.

Analysis step 2: consumes the SPE template produced by `calib/` and the
coincidence ROOT file, and delivers the deconvolved waveform in **p.e./tick**.

```bash
python deconvolve/deconvolve.py
```

## Inputs

| What | Path |
|---|---|
| SPE template | `calib/output/data/templates/<run>/template_<ch>.npy` |
| Coincidences | `calib/.roots/physics/coincident_waveforms_*.root` |

The channel → calibration run map is in `TEMPLATE_RUNS` (2070/2071 ← 039365,
2080/2081 ← 039366). Set it to `"media"` to use the cross-campaign average.

## Steps

1. **`filter_and_average`** — median baseline over the first 60 ticks, cut on
   `dt_peak_ticks_save` (per-channel window), saturation cut (`peak < 10000`),
   per-channel pre/post cuts and a shape cut (residual against the normalized
   median shape). Robust average of what survives.
2. **`align_templates`** — shifts the template peak onto the average peak
   (`np.roll`) to avoid the circular offset in the FFT deconvolution.
3. **`deconvolve_average`** — `s[f] = V[f]·G[f]/H[f]` with a Gaussian filter
   `G[f] = exp(-2(π f σ)²)`, σ from `config.GAUSS_SIGMA_TICKS`.

## Outputs (`output/`)

| Path | Contents |
|---|---|
| `plots/filtering/<ch>_persistence.png` | persistence, all vs filtered |
| `plots/average/avg_<ch>.png` | robust average of the physics signal |
| `plots/align/align_<ch>.png` | template vs average overlay (shift check) |
| `plots/deconvolution/deconv_<ch>.png` | deconvolved signal (linear + log) |
| `data/avg_<ch>.npy` | robust average |
| `data/template_aligned_<ch>.npy` | template after the `np.roll` |
| **`data/deconv_<ch>.txt`** | **deconvolved waveform — consumed by `fit/`** |

The `.txt` follows the format of the templates in
`filter/templates_large_pulses/`: one value per line, `%.9e`.

## Configuration

`config.py` holds the parameters that `fit/` also needs
(`GAUSS_SIGMA_TICKS`, `TICK_NS`), free of heavy dependencies.
