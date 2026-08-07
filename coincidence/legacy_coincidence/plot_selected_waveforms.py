#!/usr/bin/env python3
"""Plot M7/M8 waveforms written by select_cosmic_candidates.cpp."""

import argparse
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sag_analysis_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import ROOT


CHANNELS = (2070, 2071, 2080, 2081)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="ROOT file containing CosmicCandidates")
    parser.add_argument(
        "-o", "--output-dir", default="plots/cosmic_candidates",
        help="Directory for generated figures",
    )
    parser.add_argument(
        "--max-overlays", type=int, default=80,
        help="Maximum faint individual traces in each summary panel",
    )
    parser.add_argument(
        "--xmax", type=int, default=1024,
        help="Last waveform sample displayed",
    )
    parser.add_argument(
        "--candidate-ids", type=int, nargs="*", default=None,
        help="Candidate IDs for the example figure; default is low/median/high charge",
    )
    return parser.parse_args()


def read_tree(filename):
    root_file = ROOT.TFile.Open(str(filename))
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"could not open {filename}")
    tree = root_file.Get("CosmicCandidates")
    if not tree:
        raise RuntimeError(f"CosmicCandidates tree not found in {filename}")

    by_channel = {channel: [] for channel in CHANNELS}
    by_candidate = defaultdict(dict)
    metadata = {}

    for row in tree:
        channel = int(row.channel)
        if channel not in by_channel:
            continue
        candidate = int(row.candidate_id)
        # Preserve the compact input representation; convert to float only
        # when subtracting the baseline for plotting.
        adc = np.fromiter(row.adc, dtype=np.int16, count=len(row.adc))
        record = {
            "adc": adc,
            "baseline": float(row.baseline),
            "amplitude": float(row.amplitude),
            "charge": float(row.charge),
            "dt": float(row.dt_cluster_ticks),
        }
        by_channel[channel].append(record)
        by_candidate[candidate][channel] = record
        metadata[candidate] = {
            "run": int(row.run),
            "subrun": int(row.subrun),
            "event": int(row.event),
            "n_modules": int(row.n_modules),
        }

    root_file.Close()
    complete = {
        candidate: records for candidate, records in by_candidate.items()
        if set(records) == set(CHANNELS)
    }
    if not complete:
        raise RuntimeError("no complete 2070/2071/2080/2081 candidates found")
    return by_channel, complete, metadata


def baseline_subtracted(records):
    lengths = {len(record["adc"]) for record in records}
    if len(lengths) != 1:
        raise RuntimeError(f"waveforms have inconsistent lengths: {sorted(lengths)}")
    return np.stack([
        record["adc"].astype(np.float32) - record["baseline"]
        for record in records
    ])


def plot_channel_summary(by_channel, output, max_overlays, xmax):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    colors = {2070: "tab:blue", 2071: "tab:orange", 2080: "tab:green", 2081: "tab:red"}

    for axis, channel in zip(axes.flat, CHANNELS):
        waveforms = baseline_subtracted(by_channel[channel])
        stop = min(xmax, waveforms.shape[1])
        ticks = np.arange(stop)

        count = min(max_overlays, len(waveforms))
        indices = np.linspace(0, len(waveforms) - 1, count, dtype=int)
        for waveform in waveforms[indices, :stop]:
            axis.plot(ticks, waveform, color="0.45", alpha=0.07, linewidth=0.6)

        q16, median, q84 = np.percentile(waveforms[:, :stop], [16, 50, 84], axis=0)
        color = colors[channel]
        axis.fill_between(ticks, q16, q84, color=color, alpha=0.25, label="16–84%")
        axis.plot(ticks, median, color=color, linewidth=1.8, label="median")
        axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
        axis.set_title(f"Channel {channel} — {len(waveforms):,} selected waveforms")
        axis.set_ylabel("ADC − baseline")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize=9)

    for axis in axes[-1]:
        axis.set_xlabel("Sample")
    fig.suptitle("Selected M7/M8 cosmic-candidate waveforms", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def choose_candidates(by_candidate, requested):
    if requested:
        missing = [candidate for candidate in requested if candidate not in by_candidate]
        if missing:
            raise RuntimeError(f"candidate IDs not found or incomplete: {missing}")
        return requested

    scored = []
    for candidate, records in by_candidate.items():
        total_charge = sum(records[channel]["charge"] for channel in CHANNELS)
        scored.append((total_charge, candidate))
    scored.sort()
    positions = (0.10, 0.50, 0.90)
    return [scored[round(position * (len(scored) - 1))][1] for position in positions]


def plot_candidate_examples(by_candidate, metadata, candidate_ids, output, xmax):
    fig, axes = plt.subplots(
        len(candidate_ids), len(CHANNELS),
        figsize=(15, 3.2 * len(candidate_ids)), squeeze=False, sharex=True,
    )

    for row_index, candidate in enumerate(candidate_ids):
        records = by_candidate[candidate]
        meta = metadata[candidate]
        for column, channel in enumerate(CHANNELS):
            axis = axes[row_index, column]
            record = records[channel]
            waveform = record["adc"].astype(float) - record["baseline"]
            stop = min(xmax, len(waveform))
            axis.plot(np.arange(stop), waveform[:stop], linewidth=1.1)
            axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
            axis.set_title(
                f"Ch {channel}: A={record['amplitude']:.0f} ADC, "
                f"Δt={record['dt']:+.1f}"
            )
            axis.grid(alpha=0.2)
            if column == 0:
                axis.set_ylabel(
                    f"Candidate {candidate}\n"
                    f"event {meta['event']}, {meta['n_modules']} modules\nADC − baseline"
                )
            if row_index == len(candidate_ids) - 1:
                axis.set_xlabel("Sample")

    fig.suptitle("Representative complete M7/M8 coincidence candidates", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_channel, by_candidate, metadata = read_tree(args.input)

    summary = output_dir / "selected_waveforms_by_channel.png"
    examples = output_dir / "selected_candidate_examples.png"
    candidate_ids = choose_candidates(by_candidate, args.candidate_ids)
    plot_channel_summary(by_channel, summary, args.max_overlays, args.xmax)
    plot_candidate_examples(by_candidate, metadata, candidate_ids, examples, args.xmax)

    counts = ", ".join(
        f"{channel}: {len(by_channel[channel])}" for channel in CHANNELS
    )
    print(f"Complete candidates: {len(by_candidate)}")
    print(f"Waveforms per channel: {counts}")
    print(f"Example candidate IDs: {candidate_ids}")
    print(summary)
    print(examples)


if __name__ == "__main__":
    main()
