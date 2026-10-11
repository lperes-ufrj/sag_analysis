"""Fit every electric-field point of the scan in one go.

Mirrors deconvolve/all_eletric_field.py, but on the fit side: it loads the
deconvolved waveforms of every run of a branch and fits them all, writing a
single fit_results.txt with one row per (array, run).

Usage:
    python fit/all_eletric_field.py [template] [column] [branch]

    template  "mine" or "francesca"          (asked if omitted)
    column    "mean", "median", "mode" or "both"  (asked if omitted)
    branch    "up" (default), "down" or "all"

Outputs under fit/output/:
    plots/<template>/<column>/fit_<array>_run<run>.png
    data/<template>/<column>/fit_results.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calib.aux import paths
from calib.aux.runs import runs_of
from fit.fit import (COLUMNS, TEMPLATES, ask, fit_deconvolved,
                     load_deconvolved)

FIT_DIR = paths.FIT_DIR


def fit_branch(template, column, branch):
    print(f"\n=== template={template}  column={column}  branch={branch} ===")

    deconvolved = {}
    n_ok = n_missing = 0
    for kv, run in runs_of(branch):
        try:
            found = load_deconvolved(template, column, run)
        except FileNotFoundError as error:
            print(f"[MISSING] {kv:>6} kV  run {run}  -> {error}")
            n_missing += 1
            continue
        deconvolved.update(found)
        print(f"[OK]      {kv:>6} kV  run {run}  ({len(found)} waveform(s))")
        n_ok += 1

    print(f"\n{n_ok} runs loaded, {n_missing} missing "
          f"-> {len(deconvolved)} waveforms to fit\n")
    if not deconvolved:
        return None

    outdir_plots = paths.out(FIT_DIR, "plots", template, column)
    outdir_data  = paths.out(FIT_DIR, "data", template, column)
    return fit_deconvolved(deconvolved, outdir_plots, outdir_data)


def main():
    argv = sys.argv[1:]
    template = ask("SPE template used in the deconvolution", TEMPLATES,
                   argv[0] if len(argv) > 0 else None)
    column = ask("Coincidence aggregate to fit", COLUMNS + ("both",),
                 argv[1] if len(argv) > 1 else None)
    branch = argv[2] if len(argv) > 2 else "up"

    columns = ("mean", "median") if column == "both" else (column,)
    for col in columns:
        fit_branch(template, col, branch)

    print("\nNext: python fit/fit_vs_field.py " + template)
    return 0


if __name__ == "__main__":
    sys.exit(main())
