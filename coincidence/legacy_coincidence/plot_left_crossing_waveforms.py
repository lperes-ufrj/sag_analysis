#!/usr/bin/env python3
"""Plot full M7/M8 waveforms selected by select_left_crossing_cosmics.cpp."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sag_analysis_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import ROOT


CHANNELS = (2070, 2071, 2080, 2081)
COLORS = {
    2070: "tab:blue",
    2071: "tab:orange",
    2080: "tab:green",
    2081: "tab:red",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="ROOT file containing SelectedWaveforms")
    parser.add_argument(
        "-o", "--output",
        default="plots/left_crossing_cosmics/waveforms_by_channel_full.png",
        help="Output image",
    )
    parser.add_argument(
        "--mean-output",
        default="plots/left_crossing_cosmics/mean_waveforms.txt",
        help="Combined text file containing the mean waveform of each channel",
    )
    parser.add_argument(
        "--max-overlays", type=int, default=100,
        help="Maximum faint individual traces per channel",
    )
    return parser.parse_args()


def read_waveforms(filename):
    root_file = ROOT.TFile.Open(str(filename))
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"could not open {filename}")
    tree = root_file.Get("SelectedWaveforms")
    if not tree:
        raise RuntimeError(f"SelectedWaveforms tree not found in {filename}")

    waveforms = {channel: [] for channel in CHANNELS}
    for row in tree:
        channel = int(row.channel)
        if channel not in waveforms:
            continue
        adc = np.fromiter(row.adc, dtype=np.int16, count=len(row.adc))
        # Use the baseline calculated and stored by the C++ selector.
        waveforms[channel].append(
            adc.astype(np.float32) - float(row.baseline)
        )
    root_file.Close()

    result = {}
    for channel, channel_waveforms in waveforms.items():
        if not channel_waveforms:
            result[channel] = np.empty((0, 1024), dtype=np.float32)
            continue
        lengths = {len(waveform) for waveform in channel_waveforms}
        if len(lengths) != 1:
            raise RuntimeError(
                f"channel {channel} has inconsistent waveform lengths: {sorted(lengths)}"
            )
        result[channel] = np.stack(channel_waveforms)
    return result


def calculate_means(waveforms):
    means = {}
    for channel in CHANNELS:
        values = waveforms[channel]
        means[channel] = (
            np.mean(values, axis=0) if len(values) else np.full(1024, np.nan)
        )
    return means


def save_means(means, counts, output):
    lengths = {len(mean) for mean in means.values()}
    if len(lengths) != 1:
        raise RuntimeError(f"mean waveforms have inconsistent lengths: {sorted(lengths)}")
    samples = np.arange(next(iter(lengths)), dtype=int)
    table = np.column_stack([samples] + [means[channel] for channel in CHANNELS])
    count_text = ", ".join(f"N{channel}={counts[channel]}" for channel in CHANNELS)
    header = (
        "Baseline-subtracted arithmetic mean waveforms\n"
        f"{count_text}\n"
        "sample mean_2070 mean_2071 mean_2080 mean_2081"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, table, fmt=["%d", "%.8g", "%.8g", "%.8g", "%.8g"], header=header)


def plot(waveforms, means, output, max_overlays):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    for axis, channel in zip(axes.flat, CHANNELS):
        values = waveforms[channel]
        if len(values) == 0:
            axis.text(0.5, 0.5, "No selected waveforms", ha="center", va="center")
            axis.set_title(f"Channel {channel} — 0 waveforms")
            continue

        samples = np.arange(values.shape[1])
        count = min(max_overlays, len(values))
        indices = np.linspace(0, len(values) - 1, count, dtype=int)
        for waveform in values[indices]:
            axis.plot(samples, waveform, color="0.45", alpha=0.06, linewidth=0.55)

        q16, median, q84 = np.percentile(values, [16, 50, 84], axis=0)
        color = COLORS[channel]
        axis.fill_between(samples, q16, q84, color=color, alpha=0.25, label="16–84%")
        axis.plot(samples, median, color=color, linewidth=1.8, label="median")
        axis.plot(
            samples, means[channel], color="black", linestyle="--",
            linewidth=1.35, label="mean",
        )
        axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
        axis.set_xlim(0, values.shape[1] - 1)
        axis.set_title(f"Channel {channel} — {len(values):,} waveforms")
        axis.set_ylabel("ADC − baseline")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize=9)

    for axis in axes[-1]:
        axis.set_xlabel("Sample")
    fig.suptitle(
        "M7/M8 waveforms associated with independent left-crossing tags",
        fontsize=15,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    output = Path(args.output)
    mean_output = Path(args.mean_output)
    waveforms = read_waveforms(args.input)
    means = calculate_means(waveforms)
    counts = {channel: len(waveforms[channel]) for channel in CHANNELS}
    plot(waveforms, means, output, args.max_overlays)
    save_means(means, counts, mean_output)
    print("Waveforms per channel:")
    for channel in CHANNELS:
        print(f"  {channel}: {len(waveforms[channel])}")
    print(output)
    print(mean_output)


if __name__ == "__main__":
    main()
