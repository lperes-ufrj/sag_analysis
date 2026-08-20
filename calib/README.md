# calib

SPE calibration of the X-ARAPUCAs M7 and M8 from DAPHNE waveforms (PD-VD).

This folder does **calibration only**: per-waveform metrics (C++) → fingerplot
→ multigauss fit → SPE template. The deconvolution and the fit live in
`deconvolve/` and `fit/`, at the repository root (see `../README.md`).

## Channels

LArSoft channel map, code `Y0NX` (Y = module type, 1 = C / 2 = M; N = module
number; X = SiPM array, 0 or 1) → `channel = 2000 + N*10 + X`:

| Channel | Module | Window |
|---|---|---|
| 2070, 2071 | M7 | quartz, pTP inside |
| 2080, 2081 | M8 | glass, pTP coated |

---

## Step 0 — `read_waveforms.cpp`

```bash
make            # clang++ read_waveforms.cpp aux/waveform_utils.cpp -o read_waveforms $(root-config --cflags --libs)
./read_waveforms   # asks for the run number on stdin
```

Reads `.roots/np02vd_raw_run<run>*_gallery.root`, computes the metrics of each
waveform and writes **two** files:

| Output | Contents |
|---|---|
| `.roots/metrics_raw/run<run>_metrics_raw.root` | one `Metrics_<ch>` TTree per channel, **all** waveforms |
| `.roots/metrics_select/run<run>_metrics_select.root` | same, only those passing the preselection, **and with the `wf_nobaseline` branch** |

`metrics_select` is what feeds the fingerplot — i.e. **the fingerplot already
comes with the C++ preselection applied**:

```
noise    < 20  &&
sig_max  < 150 && sig_min  > -50 &&
pre_max  < 50  && pre_min  > -40 &&
post_max < 50  && post_min > -40
```

It also saves the raw and selected persistences under
`output/plots/adc-baseline/`.

> The processed channels are hardcoded in `CHANNELS[]` (currently
> `{2070, 2071}`). Switching module requires editing and recompiling — that is
> why M7 and M8 have separate calibration runs.

### Input: `*_gallery.root`

Flat ROOT files produced by the gallery pipeline. Each file holds a
`WaveformTree` with one entry per waveform:

| Branch | Type | Description |
|---|---|---|
| `run`, `subrun`, `event` | `int` | event identifiers |
| `waveform_index` | `int` | index of the waveform within the event |
| `channel` | `uint` | DAPHNE channel number |
| `timestamp` | `ull` | hardware timestamp |
| `nsamples` | `uint` | number of ADC samples |
| `adc` | `vector<short>` | raw ADC samples |

### Output: per-waveform metrics

All amplitudes and integrals are **baseline-subtracted**.

| Branch | Parameters | Description |
|---|---|---|
| `mode` | — | mode of the raw ADC samples |
| `baseline` | c=8, t=12 | mode of the waveform, then a running average over the samples within [mode±c]; after each accepted sample, skip t ticks |
| `noise` | n=240 | standard deviation over the first 240 ticks |
| `integral` | [255, 265] | sum of the samples between ticks 255 and 265 |
| `pre_min/max` | [0, 239] | min/max over the first 240 ticks (pre-signal region) |
| `sig_min/max` | [270, 1023] | min/max in the signal region |
| `post_min/max` | [350, 1023] | min/max after the peak |

---

## Steps 1 and 2 — `main.py`

```bash
python calib/main.py    # asks for the calibration run
```

A single call does both, on top of `metrics_select`:

1. **Fingerplot + multigauss fit** (`aux/fit_fingerplot.py`) — histogram of
   `integral`, fit of a pedestal Gaussian plus 3 p.e. peaks with a common gain.
   Returns `{channel: Minuit}`; the SPE gain is the `gain` parameter.
2. **SPE template** (`aux/spe_template.py`) — selects the waveforms under the
   1 to 3 p.e. peaks (window `q0 + n·gain ± σ_n`), normalizes by `n`, applies
   the quality cuts and takes the per-tick robust average.

### Quality cuts (`CUT_STAGES`)

Cumulative stages; the cutflow is printed per channel.

| Stage | Cuts |
|---|---|
| `noise` | `noise < 15` |
| `amplitude` | `sig_max < 60`, `sig_min > -50` |
| `preamplitude` | `pre_max < 35`, `pre_min > -40` |
| `postamplitude` | `post_max < 50`, `post_min > -40` |
| `tight_prepost` | `pre_max < 12.5`, `post_max < 16` |

> **Note:** `sig_min`, `pre_min` and `post_min` (and `post_max < 50`) repeat
> exactly the thresholds the C++ already applied when building
> `metrics_select` — they are no-ops. What actually cuts here is `noise < 15`,
> the upper limits on `sig_max`/`pre_max` and the `tight_prepost` stage
> (anti light-noise, ~4σ of pure noise).

### Robust average

A plain mean does not work (light noise). Per tick: median → σ = 0.5 · std
about it → mean of the values within [median ± σ] only. Median rather than
mode because, in 1 ADC bins, the mode flips between discrete levels whenever
the baseline falls between two ADC counts, injecting ~1 ADC of quantization
noise.

Finally the median of the 200 pre-signal ticks is subtracted, to compensate the
bias of the mode-based baseline estimate done in C++.

---

## Extra: `compare_templates.py`

```bash
python calib/compare_templates.py
```

Compares the SPE template of the same channel across the two calibration
campaigns (M7: 039365 vs 039473; M8: 039366 vs 039474), plots them overlaid and
saves their average under `output/data/templates/media/`. See the README in
that folder.

---

## Outputs (`output/`)

| Path | Contents |
|---|---|
| `plots/adc-baseline/` | raw and selected persistences (from the C++) |
| `plots/fingerplot/<run>/` | raw fingerplot per channel |
| `plots/multigauss/<run>/` | multigauss fit with SPE gain and χ²/ndf |
| `plots/spe_template/<run>/<ch>/` | per-peak persistences (before/after cuts) and the template |
| `plots/compare_templates/` | template comparison across campaigns |
| **`data/templates/<run>/template_<ch>.npy`** | **SPE template — consumed by `deconvolve/`** |
| `data/templates/media/` | average of the two campaigns |
| `plots/raw_wfs/` | example waveforms (C++ block currently commented out) |
| `plots/ct/` | crosstalk — orphan: `prob_crosstalk_func` exists in `aux/functions.py` but no pipeline script calls it |

## Modules

| File | Role |
|---|---|
| `main.py` | calibration runner |
| `compare_templates.py` | cross-campaign template comparison and average |
| `read_waveforms.cpp`, `aux/waveform_utils.{cpp,h}` | per-waveform metrics |
| `aux/fit_fingerplot.py` | fingerplot + multigauss fit → `{channel: Minuit}` |
| `aux/spe_template.py` | quality cuts + SPE template |
| `aux/functions.py` | `latest_cycle_keys`, `multi_gauss`, crosstalk |
| `aux/waveform.py` | `robust_template`, persistences — shared with `deconvolve/` |
| `aux/paths.py` | path anchors and `.txt` I/O |
| `aux/plot_pehist.py` | 2D histogram in p.e. (standalone, not called by `main.py`) |
