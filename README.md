# sag_analysis

Waveform analysis for the ProtoDUNE-VD (PD-VD) photon detection system.

> **Repository review (September 2026).** The original calibration documentation
> below is preserved. Some of its commands and output conventions do not match
> this checkout: `deconvolve/` and `fit/` contain no tracked implementation,
> and the current calibration entry point uses working-directory-relative paths.
> See [Current calibration behavior](#current-calibration-behavior) before
> running that pipeline. The [Repository guide](#repository-guide) covers the
> other analysis workflows and their requirements.

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

## Repository guide

The repository contains scripts, ROOT macros, C++ programs, notebooks, and
saved analysis products. There is no top-level package installer or unified
pipeline command. The main physics-analysis workflow is:

```text
Raw detector data
  → DUNE decoder + gallery extraction → *_gallery.root (WaveformTree)
  → rate analysis → reference rates / per-channel ADC thresholds
  → coincidence scan → selected-waveform identifiers in CSV
  → waveform plotting and further cuts → plots, statistics CSVs, cut summaries
  → light-yield, prompt-fraction, and electric-field studies
```

### Repository layout

| Path | Role |
|---|---|
| `selection/` | Gallery waveform extraction, decoder FHiCL configuration, and interactive event/channel inspection |
| `dir_to_tar/` | Gallery macro and FHiCL inputs for decoder job archives |
| `jobs_justin/decoder_selection_jobs/` | justIN jobs and submission scripts for calibration, cosmic, and electric-field samples |
| `set_fnal_env/` | FNAL/DUNE software, authentication, data-access, and job-input setup |
| `calib/` | C++ waveform metrics and preselection; Python SPE fits and template utilities |
| `coincidence/` | C++ coincidence scan and waveform plotting, shared analysis library, interval configuration, and input lists |
| `coincidence/coincidence_scripts_python/` | Python implementations of coincidence analysis and plotting |
| `coincidence/legacy_coincidence/` | Earlier selection tools, including the ROOT-output workflow used by `filter/filter.py` |
| `analysis/` | Rate equalization, all-channel waveform plots, and exploratory notebooks |
| `analysis/LY_vs_EF/` | Light-yield versus electric-field plots and fits, including Monte Carlo uncertainty studies |
| `filter/` | Large-pulse-template deconvolution and its reference templates |
| `src/` | Shared integration and waveform-noise readers for downstream studies |
| `plots/` | Saved analysis figures |
| `bin/` | Locally built coincidence and waveform-plotting executables; ignored by Git |

### Requirements and environment

- **Python 3** with NumPy, Matplotlib, uproot, Awkward Array, and iminuit for
  calibration and waveform utilities. Light-yield scripts also import pandas;
  the fitting scripts import `dunestyle`. `src/src.py` uses `np.trapezoid`, so
  those downstream studies require NumPy 2 or newer.
- **ROOT with PyROOT** for `analysis/rate_analysis.py`, `filter/filter.py`, and
  other ROOT-based Python tools. The active Python interpreter must be able to
  import `ROOT`.
- **A C++17 compiler, Make, and `root-config`** for compiled tools. The
  calibration Makefile selects `clang++`; the other Makefiles use `CXX` or `c++`.
- **Jupyter** for notebooks.
- **DUNE software/gallery and FNAL data access** for decoding and batch jobs.
  Already decoded gallery files can be analyzed locally with the dependencies
  above.

There is no tracked dependency manifest or lockfile. Check the active
environment before building or processing data:

```bash
root-config --version
python3 -c 'import ROOT, numpy, matplotlib, uproot, awkward, iminuit'
```

On a configured FNAL host, inspect `set_fnal_env/` before sourcing its scripts.
They assume CVMFS, UPS, and FNAL paths. In particular, `set_all.sh` sources
relative filenames and uploads a decoder-input archive with
`justin-cvmfs-upload`; it must be sourced from its own directory. The justIN
submission scripts contain explicit run lists and job settings. Adapt those
values and the output destinations before submitting jobs from
`jobs_justin/decoder_selection_jobs/`.

### Input data

The gallery macro in `selection/gallery_pdvddaphne.C` writes a `WaveformTree`
with one entry per waveform:

| Branches | Meaning |
|---|---|
| `run`, `subrun`, `event` | Event identifiers |
| `waveform_index` | Waveform index within an event |
| `channel` | DAPHNE channel number |
| `timestamp` | Hardware timestamp |
| `nsamples`, `adc` | Sample count and ADC vector |

Physics tools read text lists such as
`coincidence/input_lists/input_run039510.txt`, with one ROOT-file path per
line. Use six-digit run numbers in input-list filenames. Existing lists often
point to the author's FNAL storage; update them for your data location.
Absolute paths are the least ambiguous choice. Raw ROOT datasets are ignored
by Git and must be supplied separately.

### Build and inspect waveforms

Run the following examples from the repository root unless a command
explicitly changes directory:

```bash
make -C coincidence
make -C analysis

./bin/run_coincidence --help
./bin/plot_wfs_coincidence --help
./bin/plot_allwaveforms coincidence/input_lists/input_run039510.txt
```

The last command plots all-channel waveform persistence and saves figures to
`analysis/plots_allwaveforms/`. A Python version is also available as
`analysis/plot_allwaveforms.py`.

### Rate equalization

Choose a reference sample, then derive per-channel thresholds for the runs
being compared. These example commands use run 041523 as the reference; both
input lists must point to accessible gallery files.

```bash
python3 analysis/rate_analysis.py \
  coincidence/input_lists/input_run041523.txt --reference-sample

python3 analysis/rate_analysis.py \
  coincidence/input_lists/input_run039510.txt \
  --reference-npz analysis/RateAnalysis_data_292kV/reference_run_041523.npz
```

The script writes to `analysis/RateAnalysis_data_292kV/`: reference samples
produce `reference_run_<run>.npz`; equalized samples produce
`equalized_run_<run>.npz` and `equalized_run_<run>_thresholds.txt`.
`--freq-target-khz` can replace `--reference-npz` to specify a fixed target.
The amplitude definition uses the mean baseline over samples `[0, 50)` and
the peak over `[50, 180)`; timestamps use a 16 ns tick.

### Coincidence selection and waveform statistics

This single-run example selects M7/M8 waveforms using coincidences between
the two specified channel groups. Choose a new analysis identifier for each
set of settings, since the CSV filename contains the run and identifier.

```bash
./bin/run_coincidence coincidence/input_lists/input_run039510.txt \
  --run 039510 --timestamp example_039510 \
  --config coincidence/waveform_intervals.ini \
  --channels-coincident-left 2030 2031 2040 2041 \
  --channels-coincident-right 2050 2051 2060 2061 \
  --channels-to-save 2070 2071 2080 2081 \
  --window-ticks 10 --min-amplitude-adc 0

./bin/plot_wfs_coincidence \
  --config coincidence/waveform_intervals.ini \
  --csv coincidence/saved_coincidences/example_039510/coincidence_scan_run_039510_example_039510.csv \
  --output-dir coincidence/selected_waveforms/example_039510 \
  --max-auxiliary-amplitude 500 \
  coincidence/input_lists/input_run039510.txt
```

To apply rate equalization to the saved channels, add
`--norm-rate-adc-threshold-file analysis/RateAnalysis_data_292kV/equalized_run_039510_thresholds.txt`
to the scan command after generating the table. The scan writes a selection
CSV and a text summary under `coincidence/saved_coincidences/<identifier>/`.
The plotting stage rereads the original ROOT files, applies further waveform
cuts, and saves plots, per-channel waveform statistics, prompt-fraction CSVs,
and text cut summaries in the requested output directory. Signal, noise,
pre-signal, and post-signal intervals are configured per channel in
`coincidence/waveform_intervals.ini`.

For batches, inspect the channel choices in the wrappers, then run:

```bash
bash coincidence/run_coincidences.sh example_batch
bash coincidence/run_plot_wfs_all.sh example_batch
```

The scan wrapper discovers equalized threshold tables and reference NPZs in
`analysis/RateAnalysis_data_292kV/`, and uses matching input lists. Its current
channel groups differ from the single-run example: it saves channels
2050, 2051, 2060, and 2061. Both wrappers build their required executable.

### Large-pulse-template filtering

`filter/filter.py` provides a separate M7/M8 deconvolution workflow using
the text templates under `filter/templates_large_pulses/`. It requires a
ROOT file containing a `CoincidentWaveforms` tree with `saved_channel` and
`adc` branches, as produced by the legacy selection workflow. The current
C++ coincidence scan produces CSV selections, so its outputs are not direct
inputs to this filter.

```bash
python3 filter/filter.py /path/to/coincident_waveforms.root \
  --output-dir filter/deconvolved
```

The filter writes individual baseline-subtracted and deconvolved waveforms,
plus per-channel means, to ROOT files. Use `--help` for Gaussian-filter,
baseline, and regularization options. Template provenance and selection cuts
are documented in [the template README](filter/templates_large_pulses/README.md)
and `filter/templates_large_pulses/cuts_used.yaml`.

### Current calibration behavior

The original calibration sections above describe a layout that is only
partially represented by the current code. The following details take
precedence when running this checkout:

- `calib/main.py` opens `.roots/metrics_select/...` relative to the current
  directory, so run it from `calib/`. The reader also expects raw input files
  under `calib/.roots/`, rather than a repository-root `.roots/` directory.
- Create the reader's output directories before running it. The current
  `CHANNELS[]` selects 2070 and 2071; edit it and rebuild for an M8 pass.
- The entry point imports `calib/fit_fingerplot.py` and
  `calib/deconv_template.py`. These write plots to `calib/PLOTS/`; the latter
  returns template arrays in memory without saving `.npy` files. Utilities
  under `calib/aux/` support the newer output layout, but are not fully wired
  into this entry point. Existing saved templates do not demonstrate that
  `main.py` regenerates them.
- `deconvolve/deconvolve.py`, `deconvolve/config.py`, `fit/fit.py`, and the
  linked deconvolution/fit READMEs are absent from this checkout. The original
  full pipeline cannot run until those components are restored or implemented.
- `calib/README.md` also describes an older reader interface; the current
  executable prompts for a run number and does not accept input/output paths
  as positional arguments.

With calibration input files already placed under `calib/.roots/`, the
available metrics-and-plots workflow is:

```bash
(
  cd calib
  mkdir -p .roots/metrics_raw .roots/metrics_select plots/adc-baseline
  make
  ./read_waveforms
  python3 main.py
)
```

### Reproducibility notes

Several analysis scripts and notebooks contain fixed sample identifiers,
channel lists, or local paths; review those settings before processing a new
sample. Preserve the input lists, reference rates, interval configuration,
and cut summaries alongside derived results. Output directories vary by
workflow, so the original calibration output convention is not universal.

The tracked tree includes historical figures, selected-waveform tables,
NumPy products, compiled artifacts, and Python caches. `.gitignore` excludes
many newly generated files, but does not untrack existing artifacts. Check
`git status` before committing analysis outputs. No automated test suite or
CI configuration is included; the commands above document the source
interfaces, rather than an end-to-end validation of the physics results.
