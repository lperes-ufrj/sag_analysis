"""Deconvolve the average waveform of a coincidence selection.

Reads one aggregate CSV from coincidence/selected_waveforms/ (columns
sample, mean, median, percentile_16, percentile_84 -- the average is already
computed there) and deconvolves it with the SPE template of the channel,
using the same math as deconvolve.py.

Both the mean and the median are deconvolved and drawn together: the two
differ by ~30 % at the peak, since the coincident-amplitude distribution is
skewed and the mean is pulled by the large events.

Note: these CSVs are aggregates. The individual waveforms are in the gallery
ROOT files listed in coincidence/input_lists/ (on /pnfs), so no persistence
can be built from them -- the percentile_16/84 band is the closest proxy.

Usage:
    python deconvolve/deconvolve_coincidence.py [csv_path] [template_source]

    csv_path         path to the aggregate CSV (asked if omitted)
    template_source  a calibration run ("039365", "media", ...) or
                     "francesca" (asked if omitted)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from calib.aux import paths
from calib.aux.spe_template import (ARRAY_LABEL, load_spe_template,
                                    template_path, francesca_template_path)
from calib.aux.waveform import TICK_NS
from deconvolve.config import GAUSS_SIGMA_TICKS, MEAN_REL_ERR
from deconvolve.core import (align_template, deconvolve_fft, deconvolve_fft_err,
                             parse_coincidence_name, read_coincidence_csv)

DECONV_DIR = paths.DECONV_DIR

# default calibration run per channel, when the source is not "francesca"
DEFAULT_TEMPLATE_RUNS = {2070: "039365", 2071: "039365",
                         2080: "039366", 2081: "039366"}

COLUMNS = ("mean", "median", "mode")   # aggregates to deconvolve;
                                       # "mode" is not produced yet by
                                       # the coincidence code -- it is
                                       # skipped while absent
PLOT_XLIM_US = (0.0, 8.0)


def resolve_source(channel, source):
    """'francesca' -> itself; 'mine'/'' -> the channel's default run."""
    s = str(source).strip().lower()
    if s == "francesca":
        return "francesca"
    if s in ("", "mine", "meu", "default"):
        return DEFAULT_TEMPLATE_RUNS[channel]
    return source


def source_tag(source):
    """Output subfolder: results from the two templates never mix."""
    return "francesca" if str(source).lower() == "francesca" else "mine"


def template_label(channel, source):
    if source == "francesca":
        return f"Francesca ({francesca_template_path(channel).stem})"
    return f"this work (run {source})"


def deconvolve_coincidence(csv_path, source):
    channel, run = parse_coincidence_name(csv_path)
    source = resolve_source(channel, source)
    label = ARRAY_LABEL.get(channel, str(channel))

    tag = source_tag(source)
    outdir_plots = paths.out(DECONV_DIR, "plots", "coincidence", tag)
    outdir_data  = paths.out(DECONV_DIR, "data", "coincidence", tag)

    data = read_coincidence_csv(csv_path)
    template = load_spe_template(channel, source)

    n = data["mean"].size
    if template.size != n:
        raise ValueError(f"template has {template.size} samples, CSV has {n}")
    time_us = np.arange(n) * TICK_NS / 1000.0

    columns = [c for c in COLUMNS if c in data]
    for c in COLUMNS:
        if c not in data:
            print(f"  (column '{c}' not in the CSV yet -- skipped)")

    results = {}
    errors = {}
    for col in columns:
        v = data[col]
        tmpl_al, shift, pk_v, pk_t = align_template(template, v)
        s = deconvolve_fft(v, tmpl_al, GAUSS_SIGMA_TICKS)
        # per-sample uncertainty of the aggregate, propagated through the
        # (linear) deconvolution -- stand-in until the coincidence code
        # writes a real error column
        sigma_v = MEAN_REL_ERR * np.abs(v)
        s_err = deconvolve_fft_err(v, tmpl_al, GAUSS_SIGMA_TICKS, sigma_v)
        results[col] = s
        errors[col] = s_err
        paths.save_waveform_txt(
            outdir_data / f"deconv_{label}_run{run}_{col}.txt", s)
        paths.save_waveform_txt(
            outdir_data / f"deconv_{label}_run{run}_{col}_err.txt", s_err)
        print(f"  {label} ch{channel} run {run} [{col}]: peak {v.max():.0f} ADC "
              f"@ {pk_v}, template peak @ {pk_t} (shift {shift:+d}), "
              f"sum = {s.sum():.0f} p.e., "
              f"sigma(peak) = {s_err[int(np.argmax(s))]:.2f} p.e./tick")

    tmpl_name = template_label(channel, source)

    # three panels: input average, the SPE template actually used, result
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 11))
    fig.suptitle(f"{label} (ch {channel}) - run {run}\n"
                 f"deconvolved with SPE template: {tmpl_name}",
                 fontweight="bold")

    # 1) input: average(s) with the percentile band as a persistence proxy
    ax1.fill_between(time_us, data["percentile_16"], data["percentile_84"],
                     alpha=0.25, color="grey", label="percentile 16-84")
    for col in columns:
        ax1.plot(time_us, data[col], lw=1.0, label=col)
    ax1.set_ylabel("ADC - baseline")
    ax1.set_title("input: coincidence average", fontsize=10)
    ax1.set_xlim(*PLOT_XLIM_US)
    ax1.legend(fontsize=8)

    # 2) the template used -- its integral sets the p.e. scale, since
    #    sum(s) = V[0]/H[0] = sum(v)/sum(template)
    ax2.plot(time_us, template, lw=1.0, color="tab:green",
             label=f"{tmpl_name}\nintegral = {template.sum():.0f} ADC*tick")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("ADC - baseline")
    ax2.set_title("SPE template used", fontsize=10)
    ax2.legend(fontsize=8)

    # 3) result, with the propagated 1-sigma band
    for col in columns:
        line, = ax3.plot(time_us, results[col], lw=1.0,
                         label=f"from the {col}  ->  {results[col].sum():.0f} p.e.")
        ax3.fill_between(time_us, results[col] - errors[col],
                         results[col] + errors[col], alpha=0.25,
                         color=line.get_color(), lw=0,
                         label=f"$\\pm 1\\sigma$ ({MEAN_REL_ERR:.0%} on the {col})")
    ax3.set_xlabel("Time [us]")
    ax3.set_ylabel("p.e. / tick")
    ax3.set_title(f"deconvolved with {tmpl_name}", fontsize=10)
    ax3.set_xlim(*PLOT_XLIM_US)
    ax3.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir_plots / f"deconv_{label}_run{run}.png")
    plt.close(fig)

    return results, outdir_plots, outdir_data


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else \
        input("Path to the coincidence CSV: ").strip()
    source = sys.argv[2] if len(sys.argv) > 2 else \
        input("SPE template - 'mine' or 'francesca' [mine]: ").strip()

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        print(f"file not found: {csv_path}")
        return 1

    _, outdir_plots, outdir_data = deconvolve_coincidence(csv_path, source)
    print(f"\nPlots -> {outdir_plots}\nData  -> {outdir_data}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
