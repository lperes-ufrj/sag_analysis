import ROOT 
import numpy as np

import argparse
import re
from pathlib import Path

TICK_SECONDS = 16e-9
AMPLITUDE_DEFINITION = "baseline_to_peak"
BASELINE_RANGE = (0, 50)
SIGNAL_RANGE = (50, 180)


ROOT.gInterpreter.Declare(
    r"""
    #ifndef SAG_ANALYSIS_BASELINE_TO_PEAK_AMPLITUDE
    #define SAG_ANALYSIS_BASELINE_TO_PEAK_AMPLITUDE
    #include <algorithm>
    #include <cstddef>
    #include <limits>

    template <typename Waveform>
    double sag_mean_baseline(
        const Waveform &adc, std::size_t first, std::size_t last
    ) {
        last = std::min(last, adc.size());
        if (first >= last) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        double sum = 0.0;
        for (std::size_t sample = first; sample < last; ++sample) {
            sum += static_cast<double>(adc[sample]);
        }
        return sum / static_cast<double>(last - first);
    }

    template <typename Waveform>
    double sag_baseline_to_peak_amplitude(
        const Waveform &adc,
        double baseline,
        std::size_t first,
        std::size_t last
    ) {
        last = std::min(last, adc.size());
        if (first >= last) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const auto peak = std::max_element(
            adc.begin() + first, adc.begin() + last
        );
        return static_cast<double>(*peak) - baseline;
    }
    #endif
    """
)


def read_input_list(input_list: Path) -> tuple[str, list[Path]]:
    """Read a run input list and resolve all referenced ROOT files."""
    input_list = input_list.expanduser().resolve()
    if not input_list.is_file():
        raise FileNotFoundError(f"Input list not found: {input_list}")

    match = re.search(r"(?:input|testt)_run([0-9]{6})\.txt$", input_list.name)
    if not match:
        raise ValueError(
            "Input-list filename must have the form input_run039510.txt"
        )

    run = match.group(1)
    repo_dir = Path(__file__).resolve().parent.parent
    files = []
    for raw_line in input_list.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        listed_path = Path(line).expanduser()
        candidates = [listed_path]
        if not listed_path.is_absolute():
            candidates.insert(0, input_list.parent / listed_path)
        candidates.append(repo_dir / listed_path.name)
        path = next(
            (candidate.resolve() for candidate in candidates if candidate.is_file()),
            None,
        )
        if path is None:
            raise FileNotFoundError(
                f"ROOT file listed in {input_list} not found: {line}"
            )
        files.append(path)

    if not files:
        raise RuntimeError(f"Input list contains no ROOT files: {input_list}")
    return run, files


def build_chain(files: list[Path]) -> ROOT.TChain:
    """Build and validate a WaveformTree chain from ROOT files."""
    chain = ROOT.TChain("WaveformTree")
    for path in files:
        if chain.Add(str(path)) == 0:
            raise RuntimeError(f"Could not add ROOT file to chain: {path}")
    if chain.GetEntries() == 0:
        raise RuntimeError("WaveformTree contains no entries")
    return chain


def read_waveform_timestamps_amplitude(
    chain: ROOT.TChain,
    channels: tuple[int, ...],
) -> dict[int, dict[str, np.ndarray]]:
    """Calculate baseline-to-peak amplitude over samples 50:1023 in ROOT."""
    if not channels:
        return {}

    for branch_name in ("channel", "event", "timestamp", "adc"):
        if chain.GetBranch(branch_name) is None:
            raise RuntimeError(
                f"WaveformTree does not contain the '{branch_name}' branch"
            )

    channel_filter = " || ".join(
        f"channel == {int(channel)}" for channel in channels
    )
    data = (
        ROOT.RDataFrame(chain)
        .Filter(channel_filter)
        .Define(
            "calculated_baseline",
            "sag_mean_baseline(adc, "
            f"{BASELINE_RANGE[0]}, {BASELINE_RANGE[1]})",
        )
        .Define(
            "calculated_amplitude",
            "sag_baseline_to_peak_amplitude(adc, calculated_baseline, "
            f"{SIGNAL_RANGE[0]}, {SIGNAL_RANGE[1]})",
        )
        .AsNumpy(
            [
                "event",
                "channel",
                "timestamp",
                "calculated_amplitude",
            ]
        )
    )

    event_values = data["event"].astype(np.int64, copy=False)
    channel_values = data["channel"].astype(np.int64, copy=False)
    timestamp_values = data["timestamp"].astype(np.int64, copy=False)
    amplitude_values = data["calculated_amplitude"].astype(np.float64, copy=False)

    waveforms = {}
    for channel in channels:
        channel_mask = channel_values == channel
        waveforms[channel] = {
            "event": event_values[channel_mask],
            "timestamp": timestamp_values[channel_mask],
            "amplitude": amplitude_values[channel_mask],
        }
    return waveforms


def summarize_waveforms_by_event(
    waveforms: dict[int, dict[str, np.ndarray]],
    channels: list[int],
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Return valid amplitudes and accumulated live time for each channel."""
    amplitude_chunks = {channel: [] for channel in channels}
    live_time_s = {channel: 0.0 for channel in channels}

    for channel in channels:
        events = waveforms[channel]["event"]
        timestamps = waveforms[channel]["timestamp"]
        amplitudes = waveforms[channel]["amplitude"]
        if events.size == 0:
            continue

        order = np.argsort(events, kind="stable")
        sorted_events = events[order]
        boundaries = np.flatnonzero(np.diff(sorted_events)) + 1

        for event_indices in np.split(order, boundaries):
            selected_timestamps = timestamps[event_indices]
            if selected_timestamps.size < 3:
                continue

            event_live_time_s = (
                np.max(selected_timestamps)
                - np.min(selected_timestamps)
            ) * TICK_SECONDS
            if event_live_time_s <= 0:
                continue

            live_time_s[channel] += event_live_time_s

            selected_amplitudes = amplitudes[event_indices]
            finite = np.isfinite(selected_amplitudes)
            if np.any(finite):
                amplitude_chunks[channel].append(selected_amplitudes[finite])

    amplitude_values = {
        channel: (
            np.concatenate(amplitude_chunks[channel])
            if amplitude_chunks[channel]
            else np.empty(0, dtype=np.float64)
        )
        for channel in channels
    }
    return amplitude_values, live_time_s


def find_rate_matching_threshold(
    amplitudes: np.ndarray,
    live_time_s: float,
    target_rate_hz: float,
) -> tuple[float, np.ndarray, float]:
    """Find the amplitude threshold producing the closest rate to target."""
    if amplitudes.size == 0 or live_time_s <= 0:
        return np.nan, np.zeros(amplitudes.size, dtype=bool), np.nan
    if target_rate_hz <= 0:
        threshold = float(np.nextafter(np.max(amplitudes), np.inf))
        return threshold, np.zeros(amplitudes.size, dtype=bool), 0.0

    thresholds, counts = np.unique(amplitudes, return_counts=True)
    rates_hz = np.cumsum(counts[::-1])[::-1] / live_time_s
    best = int(np.argmin(np.abs(rates_hz - target_rate_hz)))
    threshold = float(thresholds[best])
    selected = amplitudes >= threshold
    return threshold, selected, float(np.count_nonzero(selected) / live_time_s)


def save_threshold_table(
    output: Path,
    channels: np.ndarray,
    target_frequency_khz: np.ndarray,
    thresholds_adc: np.ndarray,
) -> None:
    """Save channel, target rate, and ADC threshold as a text table."""
    if not (
        channels.shape == target_frequency_khz.shape == thresholds_adc.shape
    ):
        raise ValueError("threshold-table columns must have matching shapes")

    table = np.column_stack(
        (channels, target_frequency_khz, thresholds_adc)
    )
    np.savetxt(
        output,
        table,
        fmt=("%d", "%.4g", "%.4g"),
        delimiter="\t",
        header="channel\ttarget_frequency_khz\tthreshold_adc",
        comments="",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate per-channel baseline-to-peak amplitude spectra over "
            "waveform samples 50:1023 and save them as compressed NumPy files; "
            "equalized runs also save a text threshold table."
        )
    )
    parser.add_argument(
        "input_list",
        type=Path,
        help="text file in coincidence/input_lists containing ROOT file paths",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=6,
        help=(
            "number of ROOT worker threads (default: 6; use 0 to let ROOT "
            "choose based on available CPUs)"
        ),
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--freq-target-khz",
        "--FreqTarget",
        dest="freq_target_khz",
        type=float,
        default=None,
        help="target total waveform rate in kHz used to determine the threshold",
    )
    target_group.add_argument(
        "--reference-npz",
        type=Path,
        help=(
            "reference NPZ created with --reference-sample; its per-channel "
            "frequencies become the equalization targets"
        ),
    )
    parser.add_argument(
        "--reference-sample",
        action="store_true",
        help=(
            "mark this input as the high-electric-field reference sample; "
            "save its frequencies and histograms without thresholding"
        ),
    )
    parser.add_argument(
        "--amplitude-bin-width",
        type=float,
        default=250.0,
        help="amplitude-spectrum bin width in ADC (default: 250)",
    )
    parser.add_argument(
        "--amplitude-max",
        type=float,
        default=20000.0,
        help="maximum amplitude shown in ADC (default: 20000)",
    )

    args = parser.parse_args()
    if args.threads < 0:
        parser.error("--threads must be >= 0")
    if args.freq_target_khz is not None and args.freq_target_khz < 0:
        parser.error("--freq-target-khz must be >= 0")
    if args.amplitude_bin_width <= 0:
        parser.error("--amplitude-bin-width must be > 0")
    if args.amplitude_max <= 0:
        parser.error("--amplitude-max must be > 0")
    if args.reference_sample and (
        args.freq_target_khz is not None or args.reference_npz is not None
    ):
        parser.error(
            "--reference-sample cannot be combined with --freq-target-khz "
            "or --reference-npz"
        )
    if (
        not args.reference_sample
        and args.freq_target_khz is None
        and args.reference_npz is None
    ):
        parser.error(
            "a non-reference sample requires --reference-npz or "
            "--freq-target-khz"
        )
    return args


def main() -> None:
    args = parse_args()
    ROOT.EnableImplicitMT(args.threads)
    print(f"ROOT worker threads: {ROOT.GetThreadPoolSize()}")

    run, files = read_input_list(args.input_list)
    
    output_dir = Path(__file__).resolve().parent / "RateAnalysis_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Run: {run}")
    print(f"Input files: {len(files)}")
    for path in files:
        print(f"  {path}")

    chain = build_chain(files)

    all_channels = [1010,1011,1020,1021,1030,1031,1040,1041,1050,1051,1060,1061,1070,1071,1080,1081,2010,2011,2021,2030,2031,2040,2041,2050,2051,2060,2061,2070,2071,2080,2081]
    print(all_channels)
    waveforms = read_waveform_timestamps_amplitude(chain, all_channels)
    amplitudes, live_times = summarize_waveforms_by_event(waveforms, all_channels)

    amplitude_bins = np.arange(
        0.0,
        args.amplitude_max + args.amplitude_bin_width,
        args.amplitude_bin_width,
    )
    channels_array = np.asarray(all_channels, dtype=np.int64)
    live_time_array = np.asarray(
        [live_times[channel] for channel in all_channels], dtype=np.float64
    )
    histogram_before_counts = np.zeros(
        (len(all_channels), amplitude_bins.size - 1), dtype=np.int64
    )
    histogram_before_rate_hz = np.zeros_like(
        histogram_before_counts, dtype=np.float64
    )
    frequency_before_khz = np.full(len(all_channels), np.nan, dtype=np.float64)

    for channel_index, channel in enumerate(all_channels):
        channel_amplitudes = amplitudes[channel]
        channel_live_time = live_times[channel]
        if channel_amplitudes.size == 0 or channel_live_time <= 0:
            print(f"Channel {channel}: no valid amplitude-spectrum data")
            continue

        histogram_before_counts[channel_index], _ = np.histogram(
            channel_amplitudes, bins=amplitude_bins
        )
        histogram_before_rate_hz[channel_index] = (
            histogram_before_counts[channel_index] / channel_live_time
        )
        frequency_before_khz[channel_index] = (
            channel_amplitudes.size / channel_live_time / 1e3
        )

    common_output = {
        "run": np.asarray(run),
        "channels": channels_array,
        "amplitude_bin_edges_adc": amplitude_bins,
        "live_time_s": live_time_array,
        "amplitude_definition": np.asarray(AMPLITUDE_DEFINITION),
        "baseline_range": np.asarray(BASELINE_RANGE, dtype=np.int64),
        "signal_range": np.asarray(SIGNAL_RANGE, dtype=np.int64),
        "tick_seconds": np.asarray(TICK_SECONDS),
    }

    if args.reference_sample:
        output = output_dir / f"reference_run_{run}.npz"
        np.savez_compressed(
            output,
            **common_output,
            sample_type=np.asarray("reference"),
            frequency_khz=frequency_before_khz,
            histogram_counts=histogram_before_counts,
            histogram_rate_hz=histogram_before_rate_hz,
        )
        print(f"Saved reference data: {output}")
        return

    if args.reference_npz is not None:
        reference_path = args.reference_npz.expanduser().resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(f"Reference NPZ not found: {reference_path}")
        with np.load(reference_path, allow_pickle=False) as reference_data:
            required_keys = {
                "channels",
                "frequency_khz",
                "amplitude_definition",
                "baseline_range",
                "signal_range",
            }
            missing_keys = required_keys.difference(reference_data.files)
            if missing_keys:
                raise RuntimeError(
                    f"Reference NPZ is missing keys {sorted(missing_keys)}: "
                    f"{reference_path}. Regenerate it with this version of "
                    "rate_analysis.py."
                )
            reference_definition = str(
                reference_data["amplitude_definition"].item()
            )
            if reference_definition != AMPLITUDE_DEFINITION:
                raise RuntimeError(
                    f"Reference NPZ uses amplitude definition "
                    f"'{reference_definition}', expected "
                    f"'{AMPLITUDE_DEFINITION}': {reference_path}"
                )
            reference_baseline_range = tuple(
                reference_data["baseline_range"].astype(
                    np.int64, copy=False
                ).tolist()
            )
            if reference_baseline_range != BASELINE_RANGE:
                raise RuntimeError(
                    "Reference NPZ uses baseline range "
                    f"{reference_baseline_range}, expected {BASELINE_RANGE}: "
                    f"{reference_path}"
                )
            reference_signal_range = tuple(
                reference_data["signal_range"].astype(
                    np.int64, copy=False
                ).tolist()
            )
            if reference_signal_range != SIGNAL_RANGE:
                raise RuntimeError(
                    "Reference NPZ uses signal range "
                    f"{reference_signal_range}, expected {SIGNAL_RANGE}: "
                    f"{reference_path}"
                )
            reference_channels = reference_data["channels"].astype(
                np.int64, copy=False
            )
            reference_frequencies = reference_data["frequency_khz"].astype(
                np.float64, copy=False
            )
        reference_frequency_by_channel = dict(
            zip(reference_channels.tolist(), reference_frequencies.tolist())
        )
        missing_channels = [
            channel
            for channel in all_channels
            if channel not in reference_frequency_by_channel
        ]
        if missing_channels:
            raise RuntimeError(
                f"Reference NPZ does not contain channels: {missing_channels}"
            )
        target_frequency_khz = np.asarray(
            [reference_frequency_by_channel[channel] for channel in all_channels],
            dtype=np.float64,
        )
        target_source = str(reference_path)
    else:
        target_frequency_khz = np.full(
            len(all_channels), args.freq_target_khz, dtype=np.float64
        )
        target_source = "command_line"

    valid_target_mask = np.isfinite(target_frequency_khz)
    invalid_targets = channels_array[~valid_target_mask]
    if invalid_targets.size:
        print(
            "No finite reference frequency is available for channels: "
            f"{invalid_targets.tolist()}. Skipping these channels."
        )

        all_channels = channels_array[valid_target_mask].tolist()
        channels_array = channels_array[valid_target_mask]
        live_time_array = live_time_array[valid_target_mask]
        histogram_before_counts = histogram_before_counts[valid_target_mask]
        histogram_before_rate_hz = histogram_before_rate_hz[valid_target_mask]
        frequency_before_khz = frequency_before_khz[valid_target_mask]
        target_frequency_khz = target_frequency_khz[valid_target_mask]
        common_output["channels"] = channels_array
        common_output["live_time_s"] = live_time_array

    thresholds_adc = np.full(len(all_channels), np.nan, dtype=np.float64)
    frequency_after_khz = np.full(len(all_channels), np.nan, dtype=np.float64)
    histogram_after_counts = np.zeros_like(histogram_before_counts)
    histogram_after_rate_hz = np.zeros_like(
        histogram_before_counts, dtype=np.float64
    )

    for channel_index, channel in enumerate(all_channels):
        channel_amplitudes = amplitudes[channel]
        channel_live_time = live_times[channel]
        if channel_amplitudes.size == 0 or channel_live_time <= 0:
            continue

        threshold, selected, after_rate_hz = find_rate_matching_threshold(
            channel_amplitudes,
            channel_live_time,
            target_frequency_khz[channel_index] * 1e3,
        )
        thresholds_adc[channel_index] = threshold
        frequency_after_khz[channel_index] = after_rate_hz / 1e3
        histogram_after_counts[channel_index], _ = np.histogram(
            channel_amplitudes[selected], bins=amplitude_bins
        )
        histogram_after_rate_hz[channel_index] = (
            histogram_after_counts[channel_index] / channel_live_time
        )
        print(
            f"Channel {channel}: target={target_frequency_khz[channel_index]:.6g} "
            f"kHz, threshold={threshold:.1f} ADC, "
            f"achieved={frequency_after_khz[channel_index]:.6g} kHz"
        )

    output = output_dir / f"equalized_run_{run}.npz"
    np.savez_compressed(
        output,
        **common_output,
        sample_type=np.asarray("equalized"),
        target_source=np.asarray(target_source),
        target_frequency_khz=target_frequency_khz,
        threshold_adc=thresholds_adc,
        frequency_before_khz=frequency_before_khz,
        frequency_after_khz=frequency_after_khz,
        histogram_before_counts=histogram_before_counts,
        histogram_before_rate_hz=histogram_before_rate_hz,
        histogram_after_counts=histogram_after_counts,
        histogram_after_rate_hz=histogram_after_rate_hz,
    )
    print(f"Saved equalized data: {output}")

    threshold_output = output_dir / f"equalized_run_{run}_thresholds.txt"
    save_threshold_table(
        threshold_output,
        channels_array,
        target_frequency_khz,
        thresholds_adc,
    )
    print(f"Saved channel thresholds: {threshold_output}")
  
        
if __name__ == "__main__":
    main()
