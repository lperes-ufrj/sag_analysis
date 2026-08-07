#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from itertools import product
from pathlib import Path

import ROOT
import numpy as np

from coincidence_lib import SAMPLE_PERIOD_NS, WaveformAnalyzer, create_chain



def parse_args():
    default_config = Path(__file__).with_name("waveform_intervals.ini")

    parser = argparse.ArgumentParser(description="Test waveform coincidence")
    parser.add_argument("input_list", type=Path, help="Text file with one ROOT file per line")
    parser.add_argument("--config", default=default_config, help="Interval INI file")
    parser.add_argument(
        "--channels-coincident-left", nargs="+", type=int, default=[2030, 2031, 2040, 2041]
    )
    parser.add_argument(
        "--channels-coincident-right", nargs="+", type=int, default=[2050, 2051, 2060, 2061]
    )
    parser.add_argument(
        "--channels-to-save", nargs="+", type=int, default=[2070, 2071, 2080, 2081]
    )

    parser.add_argument("--window-ticks", type=int, default=1)
    parser.add_argument(
        "--min-amplitude-adc",
        type=float,
        default=1000.0,
        help=(
            "Minimum baseline-subtracted signal peak; waveforms must be "
            "strictly above this value (default: 1000 ADC)"
        ),
    )
    parser.add_argument("--run", required=True, help="Run number included in the output CSV filename"
    )
    
    return parser.parse_args()


def make_output_filename(args):
    left = "-".join(map(str, args.channels_coincident_left))
    right = "-".join(map(str, args.channels_coincident_right))
    save = "-".join(map(str, args.channels_to_save))
    min_amplitude = f"{args.min_amplitude_adc:g}"

    return (
        f"waveforms_run_{args.run}"
        f"_coinc_{left}_vs_{right}"
        f"_save_{save}"
        f"_window_{args.window_ticks}_ticks"
        f"_min_amplitude_{min_amplitude}_adc.csv"
    )

def read_input_list(input_list):
    files = []
    for line in input_list.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            files.append(str((input_list.parent / line).resolve()))
    return files

def scan_coincidence(
    chain,
    analyzer,
    channel_coincident_left,
    channel_coincident_right,
    channel_to_save,
    window_ticks,
    output_file="waveforms_list.csv",
    min_amplitude_adc=1000.0,
):

    chain.SetBranchStatus("*", 0)
    chain.SetBranchStatus("channel", 1)
    chain.SetBranchStatus("adc", 1)
    chain.SetBranchStatus("timestamp", 1)
    chain.SetBranchStatus("event", 1)
    chain.SetBranchStatus("waveform_index", 1)

    df = ROOT.RDataFrame(chain)
    valid_channels = np.unique(
        np.array(
            channel_coincident_left
            + channel_coincident_right
            + channel_to_save,
            dtype=np.int64,
        )
    )
    channel_evt_filter = (
        "(" + " || ".join(f"channel == {int(ch)}" for ch in valid_channels) + ")"
    )
    print("Loading filtered metadata; ADC samples will be read on demand...")
    arr = (
        df.Filter(channel_evt_filter)
        .Define("entry_index", "rdfentry_")
        .AsNumpy(
            ["entry_index", "event", "channel", "timestamp", "waveform_index"]
        )
    )

    by_event = defaultdict(lambda: defaultdict(list))
    for index, (event, channel) in enumerate(zip(arr["event"], arr["channel"])):
        by_event[int(event)][int(channel)].append(index)

    print(
        f"Loaded metadata for {len(arr['event'])} relevant waveforms "
        f"in {len(by_event)} events"
    )

    window_ns = window_ticks * SAMPLE_PERIOD_NS
    selected = {}
    total_events = len(by_event)
    coincident_events = 0
    coincident_pairs_total = 0
    subthreshold_waveforms = 0

    def pulse_time(index):
        nonlocal subthreshold_waveforms

        entry_index = int(arr["entry_index"][index])
        if chain.GetEntry(entry_index) <= 0:
            raise RuntimeError(f"Could not read chain entry {entry_index}")

        channel = int(arr["channel"][index])
        waveform = np.fromiter(
            chain.adc,
            dtype=np.int16,
            count=len(chain.adc),
        )
        amplitude = analyzer.baseline_adjusted_signal_amplitude(
            waveform, channel
        )

        if amplitude <= min_amplitude_adc:
            subthreshold_waveforms += 1
            return None

        return (
            int(arr["timestamp"][index])
            + analyzer.pulse_start(waveform, channel) * SAMPLE_PERIOD_NS
        )

    for event_number, (event, event_channels) in enumerate(by_event.items(), start=1):
        sorted_by_channel = {}

        def sorted_waveforms(channel):
            if channel in sorted_by_channel:
                return sorted_by_channel[channel]

            accepted_indices = []
            accepted_times = []
            for index in event_channels[channel]:
                time = pulse_time(int(index))
                if time is not None:
                    accepted_indices.append(index)
                    accepted_times.append(time)

            indices = np.asarray(accepted_indices, dtype=np.int64)
            times = np.asarray(accepted_times, dtype=np.int64)
            order = np.argsort(times)
            sorted_by_channel[channel] = (indices[order], times[order])
            return sorted_by_channel[channel]

        coincident_pairs = []
        event_pair_count = 0

        # Test every left/right channel pair, but only using waveforms that
        # belong to this event.
        for channel_left, channel_right in product(
            channel_coincident_left, channel_coincident_right
        ):
            left_indices, left_times = sorted_waveforms(channel_left)
            right_indices, right_times = sorted_waveforms(channel_right)

            for left_index, left_time in zip(left_indices, left_times):
                begin = np.searchsorted(
                    right_times,
                    left_time - window_ns,
                    side="left",
                )
                end = np.searchsorted(
                    right_times,
                    left_time + window_ns,
                    side="right",
                )
                matching_right_indices = right_indices[begin:end]

                if len(matching_right_indices) == 0:
                    continue

                event_pair_count += len(matching_right_indices)
                matching_right_times = right_times[begin:end]
                for right_index, right_time in zip(
                    matching_right_indices, matching_right_times
                ):
                    left_time_int = int(left_time)
                    right_time_int = int(right_time)
                    coincident_pairs.append(
                        (
                            min(left_time_int, right_time_int),
                            max(left_time_int, right_time_int),
                            right_time_int - left_time_int,
                            int(channel_left),
                            int(arr["waveform_index"][left_index]),
                            int(channel_right),
                            int(arr["waveform_index"][right_index]),
                        )
                    )

        if event_pair_count:
            coincident_events += 1
            coincident_pairs_total += event_pair_count

        coincident_pairs.sort(key=lambda pair: pair[0])
        pair_min_times = np.asarray(
            [pair[0] for pair in coincident_pairs], dtype=np.int64
        )
        pair_max_times = np.asarray(
            [pair[1] for pair in coincident_pairs], dtype=np.int64
        )
        pair_dt_times = np.asarray(
            [pair[2] for pair in coincident_pairs], dtype=np.int64
        )

        if len(coincident_pairs):
            for save_channel in channel_to_save:
                save_indices, save_times = sorted_waveforms(save_channel)

                for save_index, save_time in zip(save_indices, save_times):
                    # A coincidence pair is at most window_ns wide. Therefore,
                    # only pairs whose smallest time is in this range can be
                    # close enough to the saved waveform.
                    begin = np.searchsorted(
                        pair_min_times,
                        save_time - 2 * window_ns,
                        side="left",
                    )
                    end = np.searchsorted(
                        pair_min_times,
                        save_time + window_ns,
                        side="right",
                    )

                    for pair_index in range(begin, end):
                        if pair_max_times[pair_index] < save_time - window_ns:
                            continue

                        dt_ns_coinc = int(pair_dt_times[pair_index])
                        dt_ns_save = int(
                            save_time - pair_min_times[pair_index]
                        )
                        pair = coincident_pairs[pair_index]
                        waveform_key = (
                            event,
                            save_channel,
                            int(arr["waveform_index"][save_index]),
                        )
                        candidate = (
                            dt_ns_coinc,
                            dt_ns_save,
                            pair[3],  # coincident left channel
                            pair[4],  # coincident left waveform index
                            pair[5],  # coincident right channel
                            pair[6],  # coincident right waveform index
                        )
                        previous = selected.get(waveform_key)

                        if previous is None or (
                            abs(candidate[1]),
                            abs(candidate[0]),
                            candidate,
                        ) < (
                            abs(previous[1]),
                            abs(previous[0]),
                            previous,
                        ):
                            selected[waveform_key] = candidate

        if event_number == 1 or event_number % 1000 == 0 or event_number == total_events:
            print(
                f"Progress: {event_number}/{total_events} events; current event={event}; "
                f"coincident events={coincident_events}; "
                f"coincident pairs={coincident_pairs_total}; "
                f"unique waveforms selected={len(selected)}"
            )

    with open(output_file, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "event",
                "channel_save",
                "waveform_index_save",
                "dt_ns_coinc",
                "dt_ns_save",
                "channel_coincident_left",
                "waveform_index_coincident_left",
                "channel_coincident_right",
                "waveform_index_coincident_right",
            ]
        )
        writer.writerows(
            (*waveform_key, *selected[waveform_key])
            for waveform_key in sorted(selected)
        )

    print(
        f"Rejected {subthreshold_waveforms} waveforms with "
        f"baseline-subtracted amplitude <= {min_amplitude_adc:g} ADC"
    )
    print(f"Saved {len(selected)} unique waveforms to {output_file}")
    return set(selected)


def main():
    args = parse_args()
    files = read_input_list(args.input_list)
    if not files:
        raise ValueError(f"No ROOT files found in {args.input_list}")

    analyzer = WaveformAnalyzer(args.config)
    chain = create_chain(files)
    channel_to_save = args.channels_to_save

    output_file = make_output_filename(args)

    print(f"Output file: {output_file}")

    print(f"Input files: {len(files)}")
    print(f"Entries: {chain.GetEntries()}")
    print(
        "Minimum baseline-subtracted waveform amplitude: "
        f"> {args.min_amplitude_adc:g} ADC"
    )
    print(
        f"Testing left channels {args.channels_coincident_left} against right "
        f"channels {args.channels_coincident_right}, saving channels {channel_to_save}"
    )

    scan_coincidence(
        chain,
        analyzer,
        args.channels_coincident_left,
        args.channels_coincident_right,
        channel_to_save,
        args.window_ticks,
        output_file,
        args.min_amplitude_adc,
    )


if __name__ == "__main__":
    main()
