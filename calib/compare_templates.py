"""Compare the SPE templates of the same channel taken from different
calibration runs, and save their average.

Each X-ARAPUCA was calibrated in two independent campaigns (M7: 039365 and
039473; M8: 039366 and 039474). Since the template is normalized to 1 p.e.,
the two templates of a given channel should coincide -- the difference
between them is a direct measure of the calibration stability.

Per channel, this script:
  1. loads both templates,
  2. aligns the second onto the first by the peak (np.roll) -- the trigger
     phase changes between runs,
  3. plots them overlaid (linear, log and difference),
  4. saves the average to output/data/templates/media/template_<ch>.npy.

The 'media' folder sits inside templates/ on purpose: it behaves as a
pseudo-run, so using the average in the deconvolution is just a matter of
setting TEMPLATE_RUNS in deconvolve/deconvolve.py to {2070: "media", ...}.

Usage:
    python calib/compare_templates.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from calib.aux import paths
from calib.aux.spe_template import template_path

CALIB_DIR = paths.CALIB_DIR
MEDIA_RUN = "media"          # pseudo-run name where the average is written

# channel -> (run A, run B). A is the alignment reference.
RUN_PAIRS = {
    2070: ("039365", "039473"),   # M7, array 0
    2071: ("039365", "039473"),   # M7, array 1
    2080: ("039366", "039474"),   # M8, array 0
    2081: ("039366", "039474"),   # M8, array 1
}

MODULE = {2070: "M7", 2071: "M7", 2080: "M8", 2081: "M8"}

ALIGN = True                 # align B onto the peak of A before comparing/averaging
PULSE_WINDOW = (-50, 300)    # integration window, relative to the peak
PLOT_XLIM = (200, 700)


def integral_pulse(t, peak):
    lo = max(peak + PULSE_WINDOW[0], 0)
    hi = min(peak + PULSE_WINDOW[1], t.size)
    return float(t[lo:hi].sum())


def compare_channel(ch, run_a, run_b, plot_dir, data_dir, report):
    pa, pb = template_path(run_a, ch), template_path(run_b, ch)
    for p in (pa, pb):
        if not p.is_file():
            print(f"Channel {ch}: MISSING {p} -- skipping")
            return None

    a, b = np.load(pa), np.load(pb)
    pk_a, pk_b = int(np.argmax(a)), int(np.argmax(b))

    shift = pk_a - pk_b if ALIGN else 0
    b_al = np.roll(b, shift)

    average = 0.5 * (a + b_al)
    np.save(data_dir / f"template_{ch}.npy", average)

    # metrics
    ia, ib = integral_pulse(a, pk_a), integral_pulse(b_al, pk_a)
    d_amp = 100 * (b_al.max() - a.max()) / a.max()
    d_int = 100 * (ib - ia) / ia
    rms_res = float(np.sqrt(np.mean((b_al - a)[PLOT_XLIM[0]:PLOT_XLIM[1]] ** 2)))

    report.append(
        f"{ch:5d} {MODULE[ch]:>3s} {run_a:>8s} {run_b:>8s} {pk_a:5d} {pk_b:5d} "
        f"{shift:+4d} {a.max():8.3f} {b_al.max():8.3f} {d_amp:+7.2f} "
        f"{ia:9.2f} {ib:9.2f} {d_int:+7.2f} {rms_res:8.4f}")

    # figure: linear / log / difference
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    for ax, scale in ((ax1, "linear"), (ax2, "log")):
        ax.plot(a, lw=1.0, label=f"run {run_a}  (peak @ {pk_a})")
        ax.plot(b_al, lw=1.0, alpha=0.8,
                label=f"run {run_b}  (peak @ {pk_b}, shift {shift:+d})")
        ax.plot(average, "k--", lw=1.0, alpha=0.7, label="average")
        ax.set_ylabel("ADC - baseline")
        ax.set_yscale(scale)
        if scale == "log":
            ax.set_ylim(bottom=1e-2)
        ax.legend(fontsize=8)

    ax1.set_title(f"Ch {ch} ({MODULE[ch]}) - SPE template, {run_a} vs {run_b}\n"
                  f"amplitude {d_amp:+.2f} %, integral {d_int:+.2f} %")
    ax3.plot(b_al - a, lw=1.0, color="crimson")
    ax3.axhline(0, color="k", lw=0.6)
    ax3.set_ylabel(f"{run_b} - {run_a}  [ADC]")
    ax3.set_xlabel("Sample")
    ax3.set_xlim(*PLOT_XLIM)

    plt.tight_layout()
    fig.savefig(plot_dir / f"compare_{ch}.png")
    plt.close(fig)

    print(f"Channel {ch} ({MODULE[ch]}): shift {shift:+d}, "
          f"amplitude {d_amp:+.2f} %, integral {d_int:+.2f} %, "
          f"residual RMS {rms_res:.4f} ADC")
    return average


def main():
    plot_dir = paths.out(CALIB_DIR, "plots", "compare_templates")
    data_dir = paths.out(CALIB_DIR, "data", "templates", MEDIA_RUN)

    header = (f"{'ch':>5} {'mod':>3} {'runA':>8} {'runB':>8} {'pkA':>5} "
              f"{'pkB':>5} {'shft':>4} {'ampA':>8} {'ampB':>8} {'damp%':>7} "
              f"{'intA':>9} {'intB':>9} {'dint%':>7} {'rmsRes':>8}")
    report = [header]

    for ch, (run_a, run_b) in RUN_PAIRS.items():
        compare_channel(ch, run_a, run_b, plot_dir, data_dir, report)

    (data_dir / "comparison.txt").write_text("\n".join(report) + "\n")
    print(f"\nAverages -> {data_dir}")
    print(f"Figures  -> {plot_dir}")
    print(f"Table    -> {data_dir / 'comparison.txt'}")


if __name__ == "__main__":
    main()
