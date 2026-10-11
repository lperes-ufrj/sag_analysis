"""Scan the Gaussian-filter width of the deconvolution, sigma = 1..6 ticks.

The filter G[f] = exp(-2 (pi f sigma)^2) is what keeps 1/H[f] from blowing up
the noise. Making sigma smaller recovers detail but amplifies noise; making it
larger smooths the result and washes out the fast component. This script
deconvolves the same coincidence average with every sigma so the trade-off can
be looked at directly.

Two things are worth knowing before reading the output:

  * the total number of p.e. does NOT depend on sigma. G[0] = 1, so
    sum(s) = S[0] = V[0]/H[0] whatever the filter. Only the shape changes.
  * what does change is the peak amplitude and the noise floor, so both are
    reported: the pre-signal RMS of s(t) is the noise figure of merit.

It also produces a persistence of the six deconvolved waveforms, side by side
for our template and Francesca's, so the effect of the filter can be compared
between the two on the same figure.

By default the scan itself uses Francesca's template.

Usage:
    python deconvolve/sigma_scan.py [csv_path] [template]

    csv_path   aggregate CSV from coincidence/selected_waveforms/ (asked if omitted)
    template   "francesca" (default) or a calibration run / "mine"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from calib.aux import paths
from calib.aux.spe_template import ARRAY_LABEL, load_spe_template
from calib.aux.waveform import TICK_NS, persistence_hist
from deconvolve.core import (align_template, deconvolve_fft,
                             parse_coincidence_name, read_coincidence_csv)
from deconvolve.deconvolve_coincidence import (COLUMNS, resolve_source,
                                               source_tag, template_label)

DECONV_DIR = paths.DECONV_DIR

SIGMAS = range(1, 7)          # ticks; 1 tick = 16 ns
DEFAULT_TEMPLATE = "francesca"
N_PRE = 60                    # pre-signal ticks used as the noise figure
PLOT_XLIM_US = (0.5, 4.0)


def sigma_scan(csv_path, template):
    channel, run = parse_coincidence_name(csv_path)
    source = resolve_source(channel, template)
    label = ARRAY_LABEL.get(channel, str(channel))
    tag = source_tag(source)

    outdir_plots = paths.out(DECONV_DIR, "plots", "coincidence", "sigma_scan", tag)
    outdir_data  = paths.out(DECONV_DIR, "data", "coincidence", "sigma_scan", tag)

    data = read_coincidence_csv(csv_path)
    tmpl = load_spe_template(channel, source)
    columns = [c for c in COLUMNS if c in data]
    time_us = np.arange(data["mean"].size) * TICK_NS / 1000.0

    rows = []
    fig, axes = plt.subplots(len(columns), 1, figsize=(9, 4.5 * len(columns)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(f"{label} (ch {channel}) - run {run}\n"
                 f"Gaussian filter scan, template: {template_label(channel, source)}",
                 fontweight="bold")

    for ax, col in zip(axes, columns):
        v = data[col]
        tmpl_al, _, _, _ = align_template(tmpl, v)

        for sigma in SIGMAS:
            s = deconvolve_fft(v, tmpl_al, sigma)
            paths.save_waveform_txt(
                outdir_data / f"deconv_{label}_run{run}_{col}_sigma{sigma}.txt", s)

            noise = float(np.std(s[:N_PRE]))
            ax.plot(time_us, s, lw=1.0,
                    label=f"sigma = {sigma} ticks ({sigma * TICK_NS:.0f} ns)  "
                          f"peak {s.max():.1f}  noise {noise:.2f}")
            rows.append(f"{label:>6} {run:>8} {col:>7} {sigma:>6d} "
                        f"{sigma * TICK_NS:>9.0f} {s.max():>9.3f} "
                        f"{noise:>9.3f} {s.sum():>10.2f}")

        ax.set_ylabel("p.e. / tick")
        ax.set_title(f"from the {col}", fontsize=10)
        ax.set_xlim(*PLOT_XLIM_US)
        ax.legend(fontsize=7)

    axes[-1].set_xlabel("Time [us]")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outdir_plots / f"sigma_scan_{label}_run{run}.png")
    plt.close(fig)

    return rows, outdir_plots, outdir_data


TEMPLATES_FOR_PERSISTENCE = ("mine", "francesca")
PERSIST_XLIM = (0, 300)       # samples; the pulse sits at ~90


def sigma_persistence(csv_path):
    """Persistence of the six deconvolved waveforms (sigma = 1..6), one panel
    per template x aggregate. Same input average everywhere -- only the
    template and the filter width change."""
    channel, run = parse_coincidence_name(csv_path)
    label = ARRAY_LABEL.get(channel, str(channel))
    outdir = paths.out(DECONV_DIR, "plots", "coincidence", "sigma_scan")

    data = read_coincidence_csv(csv_path)
    columns = [c for c in COLUMNS if c in data]
    rng = np.random.default_rng(42)

    stacks = {}   # (template, column) -> (n_sigma, n_samples)
    for template in TEMPLATES_FOR_PERSISTENCE:
        source = resolve_source(channel, template)
        tmpl = load_spe_template(channel, source)
        for col in columns:
            v = data[col]
            tmpl_al, _, _, _ = align_template(tmpl, v)
            stacks[(template, col)] = np.vstack(
                [deconvolve_fft(v, tmpl_al, sg) for sg in SIGMAS])

    lo = min(a.min() for a in stacks.values())
    hi = max(a.max() for a in stacks.values())
    pad = 0.05 * (hi - lo)
    yrange = (lo - pad, hi + pad)

    n_rows = len(TEMPLATES_FOR_PERSISTENCE)
    n_cols = len(columns)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 4.5 * n_rows),
                             squeeze=False, sharex=True, sharey=True)
    fig.suptitle(f"{label} (ch {channel}) - run {run}\n"
                 f"persistence of the {len(list(SIGMAS))} deconvolutions "
                 f"(sigma = {min(SIGMAS)}..{max(SIGMAS)} ticks)",
                 fontweight="bold")

    for i, template in enumerate(TEMPLATES_FOR_PERSISTENCE):
        source = resolve_source(channel, template)
        for j, col in enumerate(columns):
            ax = axes[i][j]
            stack = stacks[(template, col)]
            h = persistence_hist(ax, stack, "", rng, yrange=yrange)
            # the six curves on top: the persistence alone is too sparse to read
            for sg, s_wf in zip(SIGMAS, stack):
                ax.plot(s_wf, lw=0.7, alpha=0.8, label=f"sigma {sg}")
            ax.set_title(f"{template_label(channel, source)}  |  {col}",
                         fontsize=9)
            ax.set_xlim(*PERSIST_XLIM)
            ax.set_ylabel("p.e. / tick")
            if i == 0 and j == 0:
                ax.legend(fontsize=6, ncol=2)
            fig.colorbar(h[3], ax=ax, label="Counts")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = outdir / f"persistence_{label}_run{run}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Persistence -> {out}")
    return out


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else \
        input("Path to the coincidence CSV: ").strip()
    template = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TEMPLATE

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        print(f"file not found: {csv_path}")
        return 1

    rows, outdir_plots, outdir_data = sigma_scan(csv_path, template)
    sigma_persistence(csv_path)

    header = (f"{'array':>6} {'run':>8} {'column':>7} {'sigma':>6} {'sigma_ns':>9} "
              f"{'peak':>9} {'noiseRMS':>9} {'sum_pe':>10}")
    print("\n".join([header] + rows))
    (outdir_data / "sigma_scan.txt").write_text("\n".join([header] + rows) + "\n")
    print(f"\nPlots -> {outdir_plots}\nData  -> {outdir_data}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
