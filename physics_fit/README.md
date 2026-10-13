# fit

Fit of the deconvolved signal: two exponentials ⊗ Gaussian + constant.

Analysis step 3: reads the `deconv_<ch>.txt` files produced by `deconvolve/`
and fits the LAr scintillation model.

```bash
python fit/fit.py
```

## Model

```
s(t) = N_f · EMG(t; t0, τ_f, σ) + N_s · EMG(t; t0, τ_s, σ) + C
```

Parametrized by the **areas** `N_f`, `N_s`: the fast/slow integrals (in p.e.)
come straight out of the fit, with Hesse errors, without manual propagation.
The EMG is evaluated in two branches with `erfcx` (`u ≥ 0` and `u < 0`) so it
does not overflow in the tail.

Per-tick error estimated from the fluctuation of the pre-signal region of the
deconvolved signal itself. Fit window in `FIT_WINDOW` (avoids the FFT window
edges and the template wrap region).

## Input

`deconvolve/output/data/deconv_<ch>.txt` — every channel found by the glob.

## Outputs (`output/`)

| Path | Contents |
|---|---|
| `plots/fit_<ch>.png` | data + fit (linear and log) |
| `data/fit_results.txt` | τ_f, τ_s, N_f, N_s, fast/total, σ, C, χ²/ndf, valid — one channel per line |
