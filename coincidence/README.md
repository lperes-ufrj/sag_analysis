# Peak finding without coincidence

`run_peak_finder.cpp` reads the `WaveformTree` trees in an input list, restricts
processing to the channels requested with `--channel` (all channels by default),
rejects saturated waveforms, and calls the existing `WaveformAnalyzer::findPeak`
on the nonsaturated waveforms. There is no coincidence CSV, event pairing,
channel-role restriction, rate threshold, or noise/post-signal/baseline-adjusted
amplitude cut. Each entry in the requested channels is processed once, including
entries with identical waveform identifiers in different files.

The final waveform population requires:

```text
!saturated
&& peak.found
&& !peak.primary_outside_signal_region
&& !peak.additional_outside_signal_region
```

At least one significant peak must exist, and all significant peaks must lie
inside the configured half-open signal interval. Multiple peaks inside are
allowed. Saturated waveforms and waveforms without a detected peak are retained
in the audit CSV but excluded from plots and waveform statistics. Saturation
means **any raw, unsmoothed ADC sample is greater than or equal to 16383** by
default. This test covers the entire waveform and occurs before baseline
subtraction and peak finding. Use `--saturation-adc N` to change the threshold
(an integer from 1 to 32767).

The plots show `ADC - baseline`, while the saturation threshold is applied to
raw ADC. A waveform clipped at 16383 with baseline 2782.5 therefore has a
plotted plateau at 13600.5. Different baselines produce different plateau
heights even at the same raw ADC limit. Regenerate outputs after changing the
filter; previously saved plots keep the selection used when they were created.

## Build and run

Requires ROOT, `root-config`, Make, and a C++17 compiler. From the repository root:

```bash
make -C coincidence run_peak_finder
bin/run_peak_finder coincidence/input_lists/input_run039510.txt \
  --run 039510 \
  --channel 2070 \
  --config coincidence/waveform_intervals.ini \
  --output-dir coincidence/peak_finder_results
```

For a single channel, use `--channel 2070`. Repeat the option for several
channels, e.g. `--channel 2070 --channel 2071`. Omit it to process all channels.
Unrequested channels are skipped before ADC loading and interval lookup and
do not appear in the output CSV or plots. The summary records the requested
channels and skipped-entry count. A requested channel with no entries produces
an empty CSV and a summary without selected-waveform plots.

For a different saturation threshold, for example:

```bash
bin/run_peak_finder coincidence/input_lists/input_run039510.txt \
  --run 039510 --channel 2070 --saturation-adc 16000
```

Use a list pointing to accessible ROOT files. List entries may be absolute
paths or paths relative to the list itself; blank lines and `#` comments are
ignored. Required branches are `event`, `channel`, `waveform_index`,
`timestamp`, and `adc` (`std::vector<short>`). Channels passed to the peak finder need
appropriate noise and signal intervals in the INI file. An absent/incompatible
branch or missing interval causes an error rather than silently skipping data.

`--run` is a required six-digit output label, as in `run_coincidence`; use one
run per input list. `--timestamp ID` optionally sets the analysis subdirectory;
otherwise the program generates `YYYYMMDD_HHMMSS` in local time. The default
configuration and output base resolve relative to the installed executable
(`bin/../coincidence/`). Use `waveform_intervals_xe.ini` for the Xe windows.

## Run all samples

From the repository root, process every `input_runNNNNNN.txt` list in
`coincidence/input_lists/`:

```bash
./coincidence/run_peak_finder_all.sh
```

The batch defaults to **channels 2080 and 2081**, with the raw saturation
threshold of 16383. It builds `run_peak_finder` when needed, generates one
timestamp for the whole batch, and writes all runs under
`coincidence/peak_finder_results/TIMESTAMP/`. It continues after individual
failures, prints successful/failed run lists, and returns a nonzero status if
any run failed. The `wrong_input_...` files and nested directories are excluded.

Supply an optional timestamp to name the batch, or override the channel list:

```bash
./coincidence/run_peak_finder_all.sh my_batch --channel 2070 --channel 2071
./coincidence/run_peak_finder_all.sh --all-channels
```

`--input-dir`, `--config`, `--output-dir`, and `--saturation-adc` are also
available. For Xe samples, select their input directory and matching intervals:

```bash
./coincidence/run_peak_finder_all.sh xe_batch \
  --input-dir coincidence/input_lists/xe_input_lists \
  --config coincidence/waveform_intervals_xe.ini
```

## Peak finder and statistics

The shared peak finder in `coincidence_lib.cpp` is unchanged:

- Baseline: median of the configured noise interval.
- Smoothing: `PEAK_SMOOTHING_WINDOW = 10`. Its existing indexing averages
  samples `n-5` through `n+5` inclusive (11 interior samples), truncating at edges.
- Peak threshold: `max(500 ADC, 10 * noiseRms)`, where `noiseRms` is the
  population standard deviation about the noise mean.
- Candidates: interior local maxima with `smoothed[n] > smoothed[n-1]` and
  `smoothed[n] >= smoothed[n+1]`, at or above the threshold.
- Primary peak: highest candidate; additional outside peaks are flagged when
  the primary is inside the signal window.

Plots and statistics use **unsmoothed**, noise-median-subtracted samples on
their original sample grid. They include the mean, standard error of the mean
(`s / sqrt(N)`, zero for one waveform), median, and linearly interpolated 16th
and 84th percentiles. The shaded band shows the percentile interval. Waveform
lengths must agree within each selected channel. Selected waveforms are held
in memory for the per-sample quantiles; rejected ADC vectors are not retained.
No prompt-fraction calculation or SiPM manufacturer mapping is needed.

The plotting/statistics implementation is shared with `plot_wfs_coincidence`
through `waveform_plotting.cpp/.hpp`; its original selection remains unchanged.

## Outputs

Files are written under `OUTPUT_DIR/TIMESTAMP/`. Let `LABEL` be
`peak_scan_run_RUN_TIMESTAMP`:

| File | Contents |
|---|---|
| `LABEL.csv` | One row per waveform in requested channels, including saturated and no-peak records |
| `LABEL.txt` | Provenance, channel/saturation/peak settings, totals, and per-channel counts |
| `channel_CHANNEL_LABEL.csv` | Per-sample statistics for selected waveforms |
| `all_waveforms_CHANNEL_LABEL_peak_selection.png` | Selected-waveform log-count density with mean and median |
| `mean_median_waveforms_LABEL.pdf` / `.png` | Channel panels with mean, median, and 16–84% band |

The audit CSV records the global `chain_entry`, event/channel/waveform identity,
timestamp, `saturated`, `peak_search_performed`, peak detection flag, primary
peak tick and smoothed amplitude, candidate count, the two outside-signal flags,
and `selected`. Saturated records have `saturated=1`, `peak_search_performed=0`,
and `selected=0`; their peak fields retain the no-peak defaults because peak
finding was skipped. The `no_peak` summary count includes only nonsaturated
waveforms actually passed to the peak finder. With no detected
peak, `peak_tick=-1`, `peak_amplitude_adc=0`, and `selected=0`. Boolean columns
use `0`/`1`. `waveform_index_save` refers directly to the ROOT `waveform_index`;
it does not imply coincidence selection. When the primary peak is outside,
the existing finder leaves `additional_outside_signal_region=0`.

An empty tree or zero selected waveforms produces a CSV and text summary and
exits successfully; there are no selected-waveform plots/statistics to write.
Outputs are written incrementally, so a later error may leave partial output.

## Integration checks

With PyROOT available to Python:

```bash
make -C coincidence test-peak-finder
```

Tests generate temporary ROOT fixtures and check multi-file scanning, repeated
identifiers, no-coincidence selection, smoothing/threshold boundaries, signal
window boundaries, multiple peaks, selected statistics/plots, empty populations,
missing input branches, missing channel configuration, inclusive raw-ADC
saturation boundaries, custom saturation thresholds, single/multiple channel
filters, skipped channels without configuration, and invalid option values.
