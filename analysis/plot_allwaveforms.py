#!/usr/bin/env python3
"""Plot waveform-density maps for every cathode and membrane channel."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import ROOT
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator


CHANNELS = {
    "cathode": tuple(range(1010, 1090, 10)) + tuple(range(1011, 1091, 10)),
    "membrane": tuple(range(2010, 2090, 10)) + tuple(range(2011, 2091, 10)),
}

ADC_BINS = 100
ADC_MIN = 0.0
ADC_MAX = 16384.0

# Arrange channels as 1010, 1011, 1020, 1021, ... in the 8 x 2 overview.
CHANNELS = {
    side: tuple(sorted(channels))
    for side, channels in CHANNELS.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create individual and 8x2 overview waveform-density plots for "
            "all cathode and membrane channels."
        )
    )
    parser.add_argument(
        "input_list",
        type=Path,
        help="text file in coincidence/input_lists containing ROOT file paths",
    )
    return parser.parse_args()


def read_input_list(input_list: Path) -> tuple[str, list[Path]]:
    input_list = input_list.expanduser().resolve()
    if not input_list.is_file():
        raise FileNotFoundError(f"Input list not found: {input_list}")

    match = (
    re.search(r"input_run([0-9]{6})\.txt$", input_list.name)
    or re.search(r"testt_run([0-9]{6})\.txt$", input_list.name)
    )


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
            raise FileNotFoundError(f"ROOT file listed in {input_list} not found: {line}")
        files.append(path)

    if not files:
        raise RuntimeError(f"Input list contains no ROOT files: {input_list}")
    return run, files


def build_chain(files: list[Path]) -> ROOT.TChain:
    chain = ROOT.TChain("WaveformTree")
    for path in files:
        if chain.Add(str(path)) == 0:
            raise RuntimeError(f"Could not add ROOT file to chain: {path}")
    if chain.GetEntries() == 0:
        raise RuntimeError("WaveformTree contains no entries")
    return chain


def collect_histograms(
    chain: ROOT.TChain,
    channels: tuple[int, ...],
    adc_bins: int,
    adc_min: float,
    adc_max: float,
) -> tuple[dict[int, np.ndarray], dict[int, int], int]:
    channel_set = set(channels)
    chain.SetBranchStatus("*", 0)
    chain.SetBranchStatus("channel", 1)
    chain.SetBranchStatus("adc", 1)

    n_samples = 0
    for entry in range(chain.GetEntries()):
        chain.GetEntry(entry)
        if int(chain.channel) in channel_set:
            n_samples = len(chain.adc)
            break
    if n_samples == 0:
        raise RuntimeError("No waveforms were found for the configured channels")

    sample_index = np.arange(n_samples)
    histograms = {
        channel: np.zeros((n_samples, adc_bins), dtype=np.uint32)
        for channel in channels
    }
    counts = {channel: 0 for channel in channels}

    n_entries = chain.GetEntries()
    for entry in range(n_entries):
        chain.GetEntry(entry)
        channel = int(chain.channel)
        if channel not in channel_set:
            continue

        waveform = np.asarray(chain.adc, dtype=np.float32)
        if len(waveform) != n_samples:
            raise RuntimeError(
                f"Channel {channel}, entry {entry} has {len(waveform)} samples; "
                f"expected {n_samples}"
            )
        adc_index = (
            (waveform - adc_min) * adc_bins / (adc_max - adc_min)
        ).astype(np.int32)
        np.clip(adc_index, 0, adc_bins - 1, out=adc_index)
        np.add.at(histograms[channel], (sample_index, adc_index), 1)
        counts[channel] += 1

        if (entry + 1) % 10000 == 0 or entry + 1 == n_entries:
            print(f"Processed {entry + 1}/{n_entries} tree entries", flush=True)

    return histograms, counts, n_samples


def draw_density(
    axis: plt.Axes,
    histogram: np.ndarray,
    channel: int,
    count: int,
    n_samples: int,
    adc_min: float,
    adc_max: float,
):
    image = axis.imshow(
        histogram.T,
        origin="lower",
        aspect="auto",
        extent=[0, n_samples, adc_min, adc_max],
        cmap="viridis",
        norm=LogNorm(vmin=1),
    )
    axis.xaxis.set_major_locator(MultipleLocator(100))
    axis.set_xlabel("Sample Index")
    axis.set_ylabel("ADC Value")
    axis.set_title(f"Channel {channel} (n={count} waveforms)")
    return image


def save_figure(figure: plt.Figure, stem: Path) -> None:
    for extension in ("pdf", "png"):
        output = stem.with_suffix(f".{extension}")
        figure.savefig(output, bbox_inches="tight", transparent=True)
        print(f"Saved {output}")


def plot_individual_channels(
    channels: tuple[int, ...],
    histograms: dict[int, np.ndarray],
    counts: dict[int, int],
    run: str,
    output_dir: Path,
    n_samples: int,
    adc_min: float,
    adc_max: float,
) -> None:
    for channel in channels:
        if counts[channel] == 0:
            print(f"Skipping channel {channel}: no waveforms")
            continue
        figure, axis = plt.subplots(figsize=(7, 3), dpi=200)
        image = draw_density(
            axis,
            histograms[channel],
            channel,
            counts[channel],
            n_samples,
            adc_min,
            adc_max,
        )
        figure.colorbar(image, ax=axis, label="Counts (log scale)")
        figure.tight_layout()
        save_figure(
            figure,
            output_dir / f"allwaveforms_channel_{channel}_run_{run}",
        )
        plt.close(figure)


def plot_detector_overview(
    side: str,
    channels: tuple[int, ...],
    histograms: dict[int, np.ndarray],
    counts: dict[int, int],
    run: str,
    output_dir: Path,
    n_samples: int,
    adc_min: float,
    adc_max: float,
) -> None:
    figure, axes = plt.subplots(8, 2, figsize=(18, 21))
    for channel, axis in zip(channels, axes.flat):
        if counts[channel] == 0:
            axis.set_title(f"Channel {channel} (no waveforms)")
            axis.set_axis_off()
            continue
        image = draw_density(
            axis,
            histograms[channel],
            channel,
            counts[channel],
            n_samples,
            adc_min,
            adc_max,
        )
        figure.colorbar(image, ax=axis, label="Counts (log scale)")
    figure.tight_layout()
    save_figure(
        figure,
        output_dir / f"allwaveforms_allarapucas_{side}_run_{run}",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    run, files = read_input_list(args.input_list)
    output_dir = Path(__file__).resolve().parent / "plots_allwaveforms"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run: {run}")
    print(f"Input files: {len(files)}")
    for path in files:
        print(f"  {path}")

    chain = build_chain(files)
    all_channels = tuple(
        channel for side in ("cathode", "membrane") for channel in CHANNELS[side]
    )
    histograms, counts, n_samples = collect_histograms(
        chain,
        all_channels,
        ADC_BINS,
        ADC_MIN,
        ADC_MAX,
    )
    print(f"Total waveforms collected: {sum(counts.values())}")

    for side in ("cathode", "membrane"):
        channels = CHANNELS[side]
        plot_individual_channels(
            channels,
            histograms,
            counts,
            run,
            output_dir,
            n_samples,
            ADC_MIN,
            ADC_MAX,
        )
        plot_detector_overview(
            side,
            channels,
            histograms,
            counts,
            run,
            output_dir,
            n_samples,
            ADC_MIN,
            ADC_MAX,
        )


if __name__ == "__main__":
    main()
