"""Plot the fit parameters against the electric field.

Reads the fit_results.txt written by fit/all_eletric_field.py for the mean and
for the median, maps each run to its cathode HV via calib/aux/runs.py, and
draws every fit variable -- including chi2/ndf -- as a function of the field.

Colour separates the aggregate (mean vs median); marker separates the SiPM
array (M8_1 vs M8_2).

Usage:
    python fit/fit_vs_field.py [template]

Output:
    fit/output/plots/<template>/fit_vs_field.png
    fit/output/data/<template>/fit_vs_field.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from calib.aux import paths
from calib.aux.plotstyle import use_dune_style
from calib.aux.runs import field_of
from fit.fit import TEMPLATES, ask

use_dune_style()

FIT_DIR = paths.FIT_DIR
FIT_DATA = FIT_DIR / "output" / "data"

COLUMNS = ("mean", "median")
COLOUR = {"mean": "tab:blue", "median": "tab:orange"}
MARKER = {"M8_1": "o", "M8_2": "s"}

# Hesse errors are underestimated whenever chi2/ndf >> 1 (here it is ~650:
# the model does not describe the tail, and the per-tick errors are neither
# correct nor independent). Scaling them by the Birge ratio sqrt(chi2/ndf) is
# the usual first aid and is what gets drawn.
SCALE_BY_BIRGE = True

# column name -> (index in fit_results.txt, index of its error or None)
FIELDS = {
    "tau_f_ns":  (2, 3),
    "tau_s_ns":  (4, 5),
    "N_f":       (6, 7),
    "N_s":       (8, 9),
    "fast_frac": (10, 11),
    "sigma_ns":  (12, 13),
    "C":         (14, 15),
    "chi2_ndf":  (16, None),
}
BIRGE_INDEX = 17
VALID_INDEX = 18

LABEL = {
    "tau_f_ns": r"$\tau_f$ [ns]",   "tau_s_ns": r"$\tau_s$ [ns]",
    "N_f": r"$N_f$ [p.e.]",         "N_s": r"$N_s$ [p.e.]",
    "fast_frac": "fast / total",    "sigma_ns": r"$\sigma$ [ns]",
    "C": "C [p.e./tick]",           "chi2_ndf": r"$\chi^2$/ndf",
}


def read_results(template, column):
    """[(array, run, field_kV, {var: (value, error)})] or [] if absent."""
    path = FIT_DATA / template / column / "fit_results.txt"
    if not path.is_file():
        print(f"  ({path} not found -- skipped)")
        return []

    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0] == "array":
            continue
        array, run = parts[0], parts[1]
        field = field_of(run)
        if field is None:
            print(f"  (run {run} is not in the field scan -- skipped)")
            continue
        birge = float(parts[BIRGE_INDEX]) if SCALE_BY_BIRGE else 1.0
        values = {name: (float(parts[i]),
                         float(parts[e]) * birge if e is not None else None)
                  for name, (i, e) in FIELDS.items()}
        if not int(parts[VALID_INDEX]):
            print(f"  (!! {array} run {run}: migrad did not converge)")
        out.append((array, run, field, values))
    return out


def main():
    argv = sys.argv[1:]
    template = ask("SPE template", TEMPLATES, argv[0] if argv else None)

    data = {col: read_results(template, col) for col in COLUMNS}
    if not any(data.values()):
        print("no fit_results.txt found -- run fit/all_eletric_field.py first")
        return 1

    n = len(FIELDS)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(13, 3.4 * ((n + 1) // 2)),
                             sharex=True)
    axes = axes.ravel()
    scaled = " (errors scaled by the Birge ratio)" if SCALE_BY_BIRGE else ""
    fig.suptitle(f"Fit parameters vs electric field - template: {template}{scaled}",
                 fontweight="bold")

    rows = []
    for ax, var in zip(axes, FIELDS):
        for col in COLUMNS:
            for array in sorted({a for a, _, _, _ in data[col]}):
                pts = sorted((f, v[var]) for a, _, f, v in data[col] if a == array)
                if not pts:
                    continue
                x = [f for f, _ in pts]
                y = [v for _, (v, _) in pts]
                e = [err for _, (_, err) in pts]
                # markers only: no line joining the points
                ax.errorbar(x, y, yerr=e if all(v is not None for v in e) else None,
                            fmt=MARKER[array], ms=5, ls="none", lw=1.2, capsize=2,
                            mfc="none" if array == "M8_2" else COLOUR[col],
                            color=COLOUR[col], alpha=0.9,
                            label=f"{col} {array}")
        ax.set_ylabel(LABEL[var], fontsize="large")
        ax.tick_params(labelsize="medium")

    for ax in axes[-2:]:
        ax.set_xlabel("Cathode HV [kV]", fontsize="large")
    axes[0].legend(fontsize="x-small", ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    outdir_plots = paths.out(FIT_DIR, "plots", template)
    outdir_data  = paths.out(FIT_DIR, "data", template)
    out = outdir_plots / "fit_vs_field.png"
    fig.savefig(out)
    plt.close(fig)

    header = f"{'column':>7} {'array':>6} {'run':>8} {'HV_kV':>7} " + \
             " ".join(f"{v:>10}" for v in FIELDS)
    for col in COLUMNS:
        for array, run, field, values in sorted(data[col], key=lambda r: (r[0], r[2])):
            rows.append(f"{col:>7} {array:>6} {run:>8} {field:>7} " +
                        " ".join(f"{values[v][0]:>10.4g}" for v in FIELDS))
    (outdir_data / "fit_vs_field.txt").write_text("\n".join([header] + rows) + "\n")

    print(f"\nPlot  -> {out}")
    print(f"Table -> {outdir_data / 'fit_vs_field.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
