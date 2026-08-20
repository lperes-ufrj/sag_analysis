# templates/media

Average of the SPE templates of the same channel obtained in **two independent
calibration campaigns**.

Produced by [`calib/compare_templates.py`](../../../compare_templates.py).

| Channel | Module | Run A | Run B |
|---|---|---|---|
| 2070 | M7 (array 0) | 039365 | 039473 |
| 2071 | M7 (array 1) | 039365 | 039473 |
| 2080 | M8 (array 0) | 039366 | 039474 |
| 2081 | M8 (array 1) | 039366 | 039474 |

## How it is built

1. Load `../<runA>/template_<ch>.npy` and `../<runB>/template_<ch>.npy`.
2. Align B onto the peak of A with `np.roll` — the trigger phase changes
   between runs (observed shifts: 0 to +2 ticks).
3. `average = 0.5 · (A + B_aligned)`.

Since the template is normalized to 1 p.e., the two should coincide: the
difference between them measures the **calibration stability** across
campaigns.

## Agreement between campaigns

Integral over the window [peak−50, peak+300]; residual RMS over [200, 700].

| Channel | Δ amplitude | Δ integral | RMS(B−A) |
|---|---|---|---|
| 2070 | −1.14 % | **+11.17 %** | 0.279 ADC |
| 2071 | +2.16 % | +2.68 % | 0.225 ADC |
| 2080 | +3.16 % | +4.54 % | 0.228 ADC |
| 2081 | −0.65 % | −0.55 % | 0.305 ADC |

Three of the four channels agree at the few-percent level. **2070 is the
outlier**: the peak amplitude matches (−1 %) but the integral differs by
+11 %, so the discrepancy lives in the **tail**, not in the peak. The
difference panel of
`../../../plots/compare_templates/compare_2070.png` shows it is essentially a
constant ~+0.3 ADC floor in run 039473 extending past tick 700, not extra late
light — which points to a residual baseline offset rather than a physical
difference. Worth understanding before using the average of this channel,
since a constant offset in the template distorts `H[f]` at low frequency.

Note that campaign B (039473/039474) has ~13× less statistics than campaign A
(~4,000 vs ~72,000 waveforms per channel), so the average is **not** weighted
by quality — it is a plain mean of the two templates.

## How to use it in the deconvolution

The folder behaves as a pseudo-run, so it is enough to point `TEMPLATE_RUNS`
in `deconvolve/deconvolve.py` at it:

```python
TEMPLATE_RUNS = {2070: "media", 2071: "media", 2080: "media", 2081: "media"}
```

## Files

| File | Contents |
|---|---|
| `template_<ch>.npy` | average template, 1024 samples, ADC − baseline |
| `comparison.txt` | full table: peaks, shift, amplitudes, integrals, RMS |
