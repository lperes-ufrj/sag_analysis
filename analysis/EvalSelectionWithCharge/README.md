# Inspecting the PDS coincidence selection with charge

`plot_tracks_PDS_coincSelection.C` reads the **art/keepup** file, runs the
existing `run_coincidence.cpp` scanner on its optical waveforms, and produces:

- XY, ZY and ZX projections of Pandora tracks, retaining other tracks as context.
- Separate wire–tick maps for every cryostat/TPC/plane, weighted by `gaushit`
  hit integral (ADC tick). These are reconstructed hits, including hits that
  are not associated with tracks.
- A separate channel–tick map of decompressed `raw::RawDigit` samples minus
  the stored pedestal. The default 8-tick bins contain signed mean ADC;
  use `rawTickRebin=1` for individual samples. ADC is not calibrated charge.
- PNGs and ROOT canvases, selected-waveform and window CSVs, track lengths,
  endpoints and associated-hit fractions, and the exact selection settings.

The inspected input has 18 events and the required products from
`pdvdkeepupstage1`: `pdvddaphne:daq`, `pandoraTrack` (tracks and track–hit
associations), `gaushit`, `tpcrawdecoder:daq` (raw digits), and `daq:trigger`.
The pasted event-dump log was from a different shard (`df-s03-d1`); these
product names were also checked in the requested `df-s04-d0` file itself.

## Run

Use a DUNE environment with Gallery, canvas, lardataobj, and ROOT (for example,
the same environment used for `selection/gallery_pdvddaphne.C`). Plain ROOT
without LArSoft dictionaries cannot read these products. From the repository:

```sh
root -l -b -q 'analysis/EvalSelectionWithCharge/plot_tracks_PDS_coincSelection.C()'
```

This defaults to the requested run-043374 file. **Until timing is configured,
this produces explicitly labelled full-event overviews of light-selected
events, not charge-window selections.** No timing constants are guessed.

For calibrated windows, load the macro and configure the values from the run's
clock/readout settings (the capitalized values below are placeholders):

```cpp
.L analysis/EvalSelectionWithCharge/plot_tracks_PDS_coincSelection.C+
PDSChargeOptions o;
o.tpcTickNs = TPC_SAMPLE_PERIOD_NS;
o.tickZeroRelativeToTriggerNs = READOUT_START_MINUS_TRIGGER_NS;
o.triggerTimestampUnitNs = NS_PER_TRIGGER_TIMESTAMP_UNIT;
o.maxDriftUs = MAXIMUM_DRIFT_TIME_US;
o.lightToChargeOffsetNs = 0.; // replace if a measured PDS/TPC offset is needed
o.maxDisplays = 0;           // all selected windows; default is 20
o.rawTickRebin = 8;
plot_tracks_PDS_coincSelection("np02vd_raw_run043374_0000_df-s04-d0_dw_0_20260320T161028_cosmic_keepup.root", "analysis/EvalSelectionWithCharge/output", o);
```

`tickZeroRelativeToTriggerNs` is negative if the first TPC sample precedes the
trigger. All displayed TPC channels must share the specified sample clock and
readout start. If different readouts have different offsets, a single global
calibration is insufficient. Check the actual run configuration.

The supplied [March 2025 workshop timetable](https://indico.fnal.gov/event/67488/timetable/#20250317.detailed)
requires login or an access key. Its slides could not be read in this session,
so no numerical timing constants are attributed to it. Supply the relevant
slides or the run timing values to complete this calibration. Native trigger
and optical timestamp ranges are exported in `clock_diagnostics.csv` to help
check units and origins. The trigger accessor itself returns an integer without
specifying physical units; see the [RDTimeStamp API](https://code-doc.larsoft.org/docs/latest/html/classraw_1_1RDTimeStamp.html).

## Selection and timing definition

The defaults are obtained directly from `Arguments` in `run_coincidence.cpp`:

- Left: 2030, 2031, 2040, 2041; right: 2050, 2051, 2060, 2061.
- Saved channels: 2070, 2071, 2080, 2081.
- `waveform_intervals.ini`, amplitude strictly greater than 0 ADC, no saved-channel
  normalized-rate amplitude cut. This is the executable default, not the
  `waveform_intervals_xe.ini` used by the separate xenon batch script.
- Pair separation `|tR - tL| <= 10 * 16 ns = 160 ns`.
- Saved waveform pulse time in `[min(tL,tR)-160, max(tL,tR)+160]` ns.

The scanner's pulse onset, amplitude tests, saved-waveform requirement,
deduplication and best-pair tie-breaking are reused without reimplementing them.
Thus the displayed pairs are those represented in the **final selected CSV**,
not every intermediate left/right pair. `selected_waveforms.csv` retains all
selected rows, including each pair's channel and waveform indices. Event IDs
are required to be unique within this single input file because the existing
scanner groups by event number; repeated IDs cause an explicit error.

The macro reconstructs the two pulse times from the original waveforms.
`minimum_timestamp_coinc` is the earlier waveform start, **not the light pulse
time**. Overlapping optical matching intervals are merged before display;
`n_pairs` records the number of distinct selected pairs in each merged interval.
No drift-expanded intervals are merged.

For each merged optical interval `[L,H]`, the charge-arrival interval is
`[L + offset, H + offset + maximum_drift_time]`. TPC tick coordinates are:

```text
low_tick  = (L - trigger_ns + offset - tick_zero_relative_to_trigger_ns) / tpc_tick_ns
high_tick = (H - trigger_ns + offset - tick_zero_relative_to_trigger_ns
             + max_drift_us * 1000) / tpc_tick_ns
```

The conversion preserves the upstream optical timestamp convention: integer
nanoseconds, as exported by `selection/gallery_pdvddaphne.C`. Verify that the
run's decoder uses this convention and that the trigger clock has a compatible
origin. Large absolute timestamps are subtracted before narrowing to double.
The drift time must correspond to this run's electric field and drift distance.
A 160 ns optical window alone is not a charge-arrival window.

## Interpreting the displays

Red tracks have at least one associated hit with peak time inside the charge
window; gray tracks have none. `tracks.csv` gives the fraction, length and
endpoints so a weak overlap cannot be mistaken for a unique match. Uncalibrated
overviews show all tracks in blue and leave timing-overlap fields blank.
Trajectory coordinates are the stored reconstructed coordinates; this macro
does not apply a new PDS t0 correction or assign a crossing-muon classification.
A drift-time window may contain charge from unrelated cosmics. Inspect topology
and endpoints, and establish a track/light match before measuring selection
purity. Anode/cathode boundaries and ARAPUCA geometry are not inferred here.

Outputs go into a new timestamped directory. `windows.csv` records every
selected optical interval and whether it was displayed, even if the display
limit is reached. `window=-1` in `tracks.csv` denotes an uncalibrated full-event
overview. `optical_input.root` is retained to rerun or compare the standalone
scanner. The input keepup file is opened read-only.

## Validation

The standalone coincidence executable can be rebuilt with
`make -C coincidence run_coincidence`. A ROOT-only regression test checks the
embedded scanner, inclusive coincidence boundaries, strict amplitude cut and
saved-channel matching:

```sh
root -l -b -q 'analysis/EvalSelectionWithCharge/test_coincidence_embedding.C()'
```

The Gallery display cannot be compiled or run on a plain-ROOT installation;
a real DUNE-environment run is still required to validate the charge plots.
