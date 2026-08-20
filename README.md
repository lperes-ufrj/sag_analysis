# sag_analysis

Waveform analysis for the ProtoDUNE-VD (PD-VD) photon detection system.

This README documents the **SPE calibration and deconvolution pipeline for the
X-ARAPUCAs M7 and M8** (channels 2070, 2071, 2080, 2081). The other folders
(`coincidence/`, `analysis/`, `filter/`, `selection/`, `src/`, `jobs_justin/`,
`set_fnal_env/`) belong to other parts of the analysis.

## Flow

```
  .roots/np02vd_raw_run<run>*_gallery.root        (raw gallery output)
                    │
                    │  calib/read_waveforms          [C++/ROOT]
                    │  per-waveform metrics + preselection
                    ▼
  calib/.roots/metrics_select/run<run>_metrics_select.root
                    │
                    │  calib/main.py                 [Python]
                    │  fingerplot → multigauss fit → SPE template
                    ▼
  calib/output/data/templates/<run>/template_<ch>.npy
                    │
                    │                        calib/.roots/physics/coincident_waveforms_*.root
                    │                                        │
                    │  deconvolve/deconvolve.py              │
                    │  filtering + robust average ◄──────────┘
                    │  template alignment
                    │  s[f] = V[f]·G[f]/H[f]
                    ▼
  deconvolve/output/data/deconv_<ch>.txt          (p.e./tick, one value per line)
                    │
                    │  fit/fit.py
                    │  2 exp ⊗ gauss + C, parametrized by the areas
                    ▼
  fit/output/data/fit_results.txt                 (τ_f, τ_s, N_f, N_s, χ²/ndf)
```

## How to run

```bash
cd calib && make && ./read_waveforms     # step 0 — asks for the run number
cd ..

python calib/main.py                     # step 1 — asks for the calibration run
python deconvolve/deconvolve.py          # step 2
python fit/fit.py                        # step 3
```

The three Python steps can be launched from any directory: paths are derived
from the location of the file itself (`__file__`), not from the `cwd`.
`read_waveforms` is the exception — it uses relative paths and must be run from
inside `calib/`.

Each step is independent: you can reprocess the fit alone without redoing the
deconvolution, as long as the `deconv_<ch>.txt` files exist.

> **M7 and M8 require two calibration passes.** `CHANNELS[]` is hardcoded in
> `read_waveforms.cpp`, so each module has its own calibration run (M7 ← 039365
> or 039473, M8 ← 039366 or 039474). The channel → run map lives in
> `TEMPLATE_RUNS`, in `deconvolve/deconvolve.py`.

## Extra tools

| Script | Purpose |
|---|---|
| `calib/compare_templates.py` | compares the SPE templates of the same channel across the two calibration campaigns and saves their average under `calib/output/data/templates/media/` |

## Conventions

**Outputs.** Each step writes only inside its own folder, under
`output/plots/` (PNG) and `output/data/` (arrays). Nothing writes into another
step's folder; steps communicate through files, at canonical paths.

**Step boundaries.** Each canonical path has a dedicated function, which is the
only place to change if the layout changes:

| Boundary | Function |
|---|---|
| calibration → deconvolution | `calib.aux.spe_template.template_path(run, ch)` |
| deconvolution → fit | `deconvolve.deconvolve.deconv_txt_path(ch)` |

**Text format.** The deconvolved waveform is written as `.txt`, one value per
line, `%.9e` — the same format as the templates in
`filter/templates_large_pulses/`. Helpers in `calib/aux/paths.py`
(`save_waveform_txt` / `load_waveform_txt`).

**Shared code.** The helpers live in `calib/aux/` (`functions.py`,
`waveform.py`, `paths.py`); `deconvolve/` and `fit/` import from there. The
deconvolution parameters that the fit also needs (`GAUSS_SIGMA_TICKS`,
`TICK_NS`) live in `deconvolve/config.py`, a module without heavy dependencies,
so that `fit/` does not have to import uproot.

## Per-step documentation

- [`calib/README.md`](calib/README.md) — channels, metrics, cuts, robust average
- [`deconvolve/README.md`](deconvolve/README.md) — filtering, alignment, Gaussian filter
- [`fit/README.md`](fit/README.md) — scintillation model and parametrization
