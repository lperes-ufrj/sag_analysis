#!/usr/bin/env python3
"""Deconvolve selected M7/M8 waveforms with the large-pulse templates.

For each input ``coincident_waveforms_fast_window_*.root`` file, this script:

1. reads the ``CoincidentWaveforms`` tree;
2. keeps only M7 and M8 channels;
3. subtracts the pre-signal baseline;
4. deconvolves the waveform with its channel-specific template in Fourier space;
5. applies a Gaussian low-pass filter in Fourier space; and
6. writes individual results and per-channel mean waveforms to a new ROOT file.

The regularized Fourier-domain calculation is

    L(f) = V(f) G(f) H*(f) / (|H(f)|^2 + lambda),

where V is the measured waveform, H is the large-pulse template, G is the
Gaussian filter, and lambda prevents divisions by very small template Fourier
components. Set ``--regularization 0`` to approach direct division V/H.
"""

from __future__ import annotations

import argparse
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import ROOT
except ImportError as error:
    raise SystemExit(
        "PyROOT is required. Load the ROOT environment before running this script."
    ) from error


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT_PATTERN = "coincident_waveforms_fast_window_20_ticks_coinc_adc_4000_save_adc_4000.root"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "deconvolved"
TEMPLATE_DIR = SCRIPT_DIR / "templates_large_pulses"

# Saved-channel numbers used by coincidence_cuts.cpp.
CHANNEL_NAMES = {
    2070: "M7_1",
    2071: "M7_2",
    2080: "M8_1",
    2081: "M8_2",
}

# Each selected channel must use the response measured with the corresponding
# X-ARAPUCA module and SiPM array.
TEMPLATE_FILES = {
    2070: TEMPLATE_DIR / "template_43229_M7_1.txt",
    2071: TEMPLATE_DIR / "template_43229_M7_2.txt",
    2080: TEMPLATE_DIR / "template_42321_M8_1.txt",
    2081: TEMPLATE_DIR / "template_42321_M8_2.txt",
}


@dataclass(frozen=True)
class PreparedTemplate:
    """Fourier representation and diagnostic information for one template."""

    spectrum: np.ndarray
    peak_tick: int
    baseline: float


@dataclass
class ChannelAccumulator:
    """Running sums used to calculate mean waveforms without storing entries."""

    count: int
    baseline_subtracted: np.ndarray
    deconvolved: np.ndarray
    deconvolved_filtered: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deconvolve M7/M8 waveforms selected by coincidence_cuts.cpp "
            "using channel-specific large-pulse templates."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "Input ROOT files. If omitted, all matching files in analysis/ "
            "are processed. Shell wildcards are supported by the shell."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output ROOT file; valid only when processing one input file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for automatically named outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=30,
        help="Number of initial waveform samples used for baseline estimation.",
    )
    parser.add_argument(
        "--template-baseline-samples",
        type=int,
        default=200,
        help="Number of initial template samples used for baseline estimation.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=0.05,
        help=(
            "Gaussian frequency sigma in cycles/sample; the Nyquist frequency "
            "is 0.5 cycles/sample (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0e-8,
        help=(
            "Tikhonov strength relative to max(|H|^2), used to stabilize the "
            "inverse filter (default: 1e-8)."
        ),
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()

    if args.baseline_samples <= 0:
        parser.error("--baseline-samples must be greater than zero")
    if args.template_baseline_samples <= 0:
        parser.error("--template-baseline-samples must be greater than zero")
    if not 0.0 < args.gaussian_sigma <= 0.5:
        parser.error("--gaussian-sigma must be in the interval (0, 0.5]")
    if args.regularization < 0.0:
        parser.error("--regularization must be non-negative")

    if not args.inputs:
        args.inputs = sorted((REPO_DIR / "analysis").glob(DEFAULT_INPUT_PATTERN))

    args.inputs = [path.resolve() for path in args.inputs]
    missing = [path for path in args.inputs if not path.is_file()]
    if missing:
        parser.error("input file(s) not found: " + ", ".join(map(str, missing)))
    if not args.inputs:
        parser.error(
            f"no files matching analysis/{DEFAULT_INPUT_PATTERN} were found"
        )
    if args.output is not None and len(args.inputs) != 1:
        parser.error("--output can only be used with exactly one input file")

    return args


def load_and_prepare_template(
    path: Path,
    waveform_length: int,
    baseline_samples: int,
) -> PreparedTemplate:
    """Load, baseline-correct, align, and transform a response template."""

    template = np.loadtxt(path, dtype=np.float64)
    if template.ndim != 1:
        raise ValueError(f"template must be one-dimensional: {path}")
    if template.size != waveform_length:
        raise ValueError(
            f"{path.name} has {template.size} samples, but the waveform has "
            f"{waveform_length}"
        )
    if baseline_samples > template.size:
        raise ValueError(
            f"template baseline length ({baseline_samples}) exceeds the "
            f"template length ({template.size})"
        )

    baseline = float(np.mean(template[:baseline_samples]))
    corrected = template - baseline
    peak_tick = int(np.argmax(np.abs(corrected)))

    # Move the response peak to tick zero. This removes the template's arbitrary
    # acquisition offset while retaining its phase and pulse shape.
    aligned = np.roll(corrected, -peak_tick)
    spectrum = np.fft.rfft(aligned)

    return PreparedTemplate(
        spectrum=spectrum,
        peak_tick=peak_tick,
        baseline=baseline,
    )


def gaussian_frequency_filter(length: int, sigma: float) -> np.ndarray:
    """Return G(f)=exp[-f^2/(2 sigma^2)] on the real-FFT frequency grid."""

    frequencies = np.fft.rfftfreq(length, d=1.0)
    return np.exp(-0.5 * (frequencies / sigma) ** 2)


def deconvolve(
    waveform: np.ndarray,
    template: PreparedTemplate,
    gaussian_filter: np.ndarray,
    baseline_samples: int,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Baseline-subtract and return unfiltered and filtered deconvolutions."""

    if baseline_samples > waveform.size:
        raise ValueError(
            f"waveform baseline length ({baseline_samples}) exceeds waveform "
            f"length ({waveform.size})"
        )

    baseline = float(np.mean(waveform[:baseline_samples]))
    corrected = waveform - baseline
    waveform_spectrum = np.fft.rfft(corrected)

    template_power = np.abs(template.spectrum) ** 2
    max_power = float(np.max(template_power))
    if max_power == 0.0:
        raise ValueError("the selected template has no nonzero Fourier components")

    # The numerical floor also protects direct division when regularization=0.
    floor = max(regularization * max_power, np.finfo(np.float64).eps * max_power)
    inverse_template = np.conj(template.spectrum) / (template_power + floor)
    deconvolved_spectrum = waveform_spectrum * inverse_template

    unfiltered = np.fft.irfft(deconvolved_spectrum, n=waveform.size)
    filtered = np.fft.irfft(
        deconvolved_spectrum * gaussian_filter,
        n=waveform.size,
    )
    return corrected, unfiltered, filtered, baseline


def replace_vector(target, values: np.ndarray) -> None:
    """Replace the contents of a ROOT std::vector<double>."""

    target.clear()
    target.reserve(int(values.size))
    for value in values:
        target.push_back(float(value))


def make_output_path(input_path: Path, args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output.resolve()

    output_dir = args.output_dir.resolve()
    return output_dir / f"deconvolved_{input_path.name}"


def write_mean_tree(
    output_file,
    accumulators: dict[int, ChannelAccumulator],
) -> None:
    """Write one mean baseline-corrected/deconvolved entry per channel."""

    output_file.cd()
    tree = ROOT.TTree(
        "MeanDeconvolvedWaveforms",
        "Mean M7/M8 waveforms after template deconvolution",
    )

    channel = array("i", [0])
    n_waveforms = array("q", [0])
    mean_adc = ROOT.std.vector("double")()
    mean_deconvolved = ROOT.std.vector("double")()
    mean_deconvolved_filtered = ROOT.std.vector("double")()

    tree.Branch("channel", channel, "channel/I")
    tree.Branch("n_waveforms", n_waveforms, "n_waveforms/L")
    tree.Branch("mean_adc_baseline_subtracted", mean_adc)
    tree.Branch("mean_deconvolved", mean_deconvolved)
    tree.Branch("mean_deconvolved_filtered", mean_deconvolved_filtered)

    for channel_number in sorted(accumulators):
        accumulator = accumulators[channel_number]
        channel[0] = channel_number
        n_waveforms[0] = accumulator.count
        replace_vector(
            mean_adc,
            accumulator.baseline_subtracted / accumulator.count,
        )
        replace_vector(
            mean_deconvolved,
            accumulator.deconvolved / accumulator.count,
        )
        replace_vector(
            mean_deconvolved_filtered,
            accumulator.deconvolved_filtered / accumulator.count,
        )
        tree.Fill()

    tree.Write()


def process_file(input_path: Path, output_path: Path, args: argparse.Namespace) -> int:
    """Process one coincidence file and return its number of written entries."""

    input_file = ROOT.TFile.Open(str(input_path), "READ")
    if not input_file or input_file.IsZombie():
        raise OSError(f"could not open input ROOT file: {input_path}")

    input_tree = input_file.Get("CoincidentWaveforms")
    if not input_tree:
        input_file.Close()
        raise KeyError(f"CoincidentWaveforms tree not found in {input_path}")
    if not input_tree.GetBranch("saved_channel") or not input_tree.GetBranch("adc"):
        input_file.Close()
        raise KeyError(f"saved_channel or adc branch not found in {input_path}")
    if input_tree.GetEntries() == 0:
        input_file.Close()
        raise ValueError(f"CoincidentWaveforms contains no entries: {input_path}")

    # Read one entry to establish the waveform length and validate all templates.
    input_tree.GetEntry(0)
    waveform_length = len(input_tree.adc)
    if waveform_length == 0:
        input_file.Close()
        raise ValueError(f"the first waveform is empty: {input_path}")

    templates = {
        channel: load_and_prepare_template(
            path,
            waveform_length,
            args.template_baseline_samples,
        )
        for channel, path in TEMPLATE_FILES.items()
    }
    frequency_filter = gaussian_frequency_filter(
        waveform_length,
        args.gaussian_sigma,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = ROOT.TFile(str(output_path), "RECREATE")
    if output_file.IsZombie():
        input_file.Close()
        raise OSError(f"could not create output ROOT file: {output_path}")

    output_file.cd()
    output_tree = input_tree.CloneTree(0)
    output_tree.SetName("DeconvolvedWaveforms")
    output_tree.SetTitle("M7/M8 waveforms after large-template deconvolution")

    adc_baseline_subtracted = ROOT.std.vector("double")()
    adc_deconvolved = ROOT.std.vector("double")()
    adc_deconvolved_filtered = ROOT.std.vector("double")()
    waveform_baseline = array("d", [0.0])
    deconvolved_integral = array("d", [0.0])
    deconvolved_filtered_integral = array("d", [0.0])
    template_peak_tick = array("i", [0])

    output_tree.Branch("adc_baseline_subtracted", adc_baseline_subtracted)
    output_tree.Branch("adc_deconvolved", adc_deconvolved)
    output_tree.Branch("adc_deconvolved_filtered", adc_deconvolved_filtered)
    output_tree.Branch("waveform_baseline", waveform_baseline, "waveform_baseline/D")
    output_tree.Branch(
        "deconvolved_integral",
        deconvolved_integral,
        "deconvolved_integral/D",
    )
    output_tree.Branch(
        "deconvolved_filtered_integral",
        deconvolved_filtered_integral,
        "deconvolved_filtered_integral/D",
    )
    output_tree.Branch("template_peak_tick", template_peak_tick, "template_peak_tick/I")

    accumulators: dict[int, ChannelAccumulator] = {}
    written = 0

    for entry_index in range(input_tree.GetEntries()):
        input_tree.GetEntry(entry_index)
        channel = int(input_tree.saved_channel)
        if channel not in CHANNEL_NAMES:
            continue

        waveform = np.fromiter(input_tree.adc, dtype=np.float64)
        if waveform.size != waveform_length:
            raise ValueError(
                f"entry {entry_index} in {input_path.name} has {waveform.size} "
                f"samples; expected {waveform_length}"
            )

        corrected, unfiltered, filtered, baseline = deconvolve(
            waveform,
            templates[channel],
            frequency_filter,
            args.baseline_samples,
            args.regularization,
        )

        replace_vector(adc_baseline_subtracted, corrected)
        replace_vector(adc_deconvolved, unfiltered)
        replace_vector(adc_deconvolved_filtered, filtered)
        waveform_baseline[0] = baseline
        deconvolved_integral[0] = float(np.sum(unfiltered))
        deconvolved_filtered_integral[0] = float(np.sum(filtered))
        template_peak_tick[0] = templates[channel].peak_tick
        output_tree.Fill()
        written += 1

        if channel not in accumulators:
            accumulators[channel] = ChannelAccumulator(
                count=0,
                baseline_subtracted=np.zeros(waveform_length, dtype=np.float64),
                deconvolved=np.zeros(waveform_length, dtype=np.float64),
                deconvolved_filtered=np.zeros(waveform_length, dtype=np.float64),
            )
        accumulator = accumulators[channel]
        accumulator.count += 1
        accumulator.baseline_subtracted += corrected
        accumulator.deconvolved += unfiltered
        accumulator.deconvolved_filtered += filtered

    output_file.cd()
    output_tree.Write()
    write_mean_tree(output_file, accumulators)
    output_file.Close()
    input_file.Close()

    print(f"{input_path.name}: wrote {written} M7/M8 waveforms to {output_path}")
    for channel in sorted(accumulators):
        print(f"  {CHANNEL_NAMES[channel]} ({channel}): {accumulators[channel].count}")
    return written


def main() -> int:
    args = parse_args()

    print("Channel-template mapping:")
    for channel, path in TEMPLATE_FILES.items():
        print(f"  {CHANNEL_NAMES[channel]} ({channel}): {path.name}")
    print(f"Gaussian sigma: {args.gaussian_sigma:g} cycles/sample")
    print(f"Regularization: {args.regularization:g} x max(|H|^2)")

    total_written = 0
    for input_path in args.inputs:
        output_path = make_output_path(input_path, args)
        if output_path == input_path:
            raise ValueError("the output path must differ from the input path")
        total_written += process_file(input_path, output_path, args)

    print(f"Finished: wrote {total_written} waveforms from {len(args.inputs)} file(s).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
