"""Compare the deconvolution of the same coincidence average using our SPE
template and Francesca's.

The input average is identical in both branches -- only the template changes --
so any difference in the result is attributable to the template alone.

One figure per channel, two panels (from the mean and from the median of the
coincidence aggregate), each with the two deconvolutions overlaid.

Usage:
    python deconvolve/compare_deconv_templates.py [csv_path ...]
    python deconvolve/compare_deconv_templates.py --run 039500

With no argument it asks for the run and takes every channel available for it.
"""
import sys
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from calib.aux import paths
from calib.aux.spe_template import ARRAY_LABEL, load_spe_template
from calib.aux.waveform import TICK_NS
from deconvolve.config import GAUSS_SIGMA_TICKS
from deconvolve.core import (align_template, deconvolve_fft,
                             parse_coincidence_name, read_coincidence_csv)
from deconvolve.deconvolve_coincidence import (COLUMNS, DEFAULT_TEMPLATE_RUNS,
                                               PLOT_XLIM_US, template_label)

DECONV_DIR = paths.DECONV_DIR
COINC_DIR = paths.REPO_ROOT / "coincidence" / "selected_waveforms"


def csvs_for_run(run):
    pattern = str(COINC_DIR / f"channel_*_run_{run}_*.csv")
    return sorted(Path(p) for p in glob.glob(pattern))


def compare_one(csv_path, outdir_plots, report):
    channel, run = parse_coincidence_name(csv_path)
    label = ARRAY_LABEL.get(channel, str(channel))
    data = read_coincidence_csv(csv_path)

    sources = [DEFAULT_TEMPLATE_RUNS[channel], "francesca"]
    time_us = np.arange(data["mean"].size) * TICK_NS / 1000.0

    n_col = sum(1 for c in COLUMNS if c in data)
    fig, axes = plt.subplots(n_col, 1, figsize=(9, 4 * n_col), sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(f"{label} (ch {channel}) - run {run}\n"
                 f"same average, deconvolved with two SPE templates",
                 fontweight="bold")

    columns = [c for c in COLUMNS if c in data]
    for ax, col in zip(axes, columns):
        v = data[col]
        for source in sources:
            template = load_spe_template(channel, source)
            tmpl_al, _, _, _ = align_template(template, v)
            s = deconvolve_fft(v, tmpl_al, GAUSS_SIGMA_TICKS)
            ax.plot(time_us, s, lw=1.0,
                    label=f"template: {template_label(channel, source)}\n"
                          f"   integral {template.sum():.0f} ADC*tick"
                          f"  ->  {s.sum():.0f} p.e.")
            report.append((label, channel, run, col, str(source),
                           float(s.max()), float(s.sum())))
        ax.set_ylabel("p.e. / tick")
        ax.set_title(f"from the {col}", fontsize=10)
        ax.set_xlim(*PLOT_XLIM_US)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Time [us]")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outdir_plots / f"deconv_templates_{label}_run{run}.png")
    plt.close(fig)


def main():
    args = sys.argv[1:]
    if args and args[0] == "--run":
        csvs = csvs_for_run(args[1])
    elif args:
        csvs = [Path(a) for a in args]
    else:
        csvs = csvs_for_run(input("Run number (e.g. 039500): ").strip())

    csvs = [c for c in csvs if c.is_file()]
    if not csvs:
        print("no coincidence CSV found")
        return 1

    # uses both templates, so it belongs to neither -- its own folder
    outdir_plots = paths.out(DECONV_DIR, "plots", "coincidence", "comparison")
    outdir_data  = paths.out(DECONV_DIR, "data", "coincidence", "comparison")

    report = []
    for csv_path in csvs:
        compare_one(csv_path, outdir_plots, report)

    header = f"{'array':>6} {'ch':>5} {'run':>8} {'column':>7} {'template':>10} {'peak':>10} {'sum_pe':>10}"
    lines = [header] + [
        f"{a:>6} {c:>5d} {r:>8} {col:>7} {src:>10} {pk:>10.3f} {tot:>10.1f}"
        for a, c, r, col, src, pk, tot in report]
    (outdir_data / "deconv_templates_comparison.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nPlots -> {outdir_plots}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
