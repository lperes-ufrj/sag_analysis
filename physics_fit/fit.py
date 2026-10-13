"""Fit of the deconvolved signal: 2 exponentials (x) Gaussian + C.

Analysis step 3: reads the deconvolved waveforms produced by deconvolve/
and fits the LAr scintillation model.

The input is picked by three keys, asked interactively or given on the
command line:

    template   which SPE template the deconvolution used: "mine" or
               "francesca" (the two live in sibling folders with identical
               filenames, so this must be explicit)
    column     which aggregate of the coincidence average: "mean", "median"
               or "mode" ("mode" is not produced by the coincidence code yet)
    run        a run number (e.g. 039500) or "all"

Usage:
    python fit/fit.py [template] [column] [run]
    python fit/fit.py --legacy            (the flat deconv_<ch>.txt from the
                                           ROOT pipeline of deconvolve.py)

Outputs under fit/output/:
    plots/<template>/<column>/fit_<array>_run<run>.png
    data/<template>/<column>/fit_results.txt
"""
import os
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcx
from iminuit import Minuit
from iminuit.cost import LeastSquares

from calib.aux import paths
from calib.aux.plotstyle import use_dune_style
from deconvolve.config import GAUSS_SIGMA_TICKS, MEAN_REL_ERR, TICK_NS

use_dune_style()

FIT_DIR = paths.FIT_DIR
DECONV_DATA = paths.DECONV_DIR / "output" / "data"
COINC_DATA = DECONV_DATA / "coincidence"

TEMPLATES = ("mine", "francesca")
COLUMNS = ("mean", "median", "mode")

# tau_f is not resolvable here: at 16 ns/tick, and after the sigma = 6 ticks
# (96 ns) Gaussian filter of the deconvolution, the fast component is a delta.
# Left free, migrad returns tau_f = 0.1 +- 2.6 ticks sitting on its limit and
# the minimum is flagged INVALID (flat direction). Setting this to a number
# fixes tau_f [ns] to that value, which makes N_f, N_s and the fast fraction
# well defined; None reproduces the free-tau_f fit.
FIX_TAU_F_NS = float(os.environ["FIT_FIX_TAU_F"]) \
    if os.environ.get("FIT_FIX_TAU_F") else None

# fit window on the deconvolved signal (ticks): avoids the FFT window edges
# and the template wrap region (~845).
#
# The upper edge matters a lot. The coincidence average v(t) does not return
# to its pre-signal baseline: it crosses zero around tick 430 and settles on a
# plateau of about -4 % of the peak (-152 ADC out of 3561 at 0 kV). The SPE
# template has no matching undershoot, so the deconvolved s(t) inherits that
# step, the fit absorbs it into C, and tau_s is dragged down -- from ~1130 ns
# with an upper edge of 200 ticks to ~950 ns at 700 ticks (see the window scan
# in the analysis notes). chi2/ndf is minimal around 300 ticks, where C is
# still small and stable; that is the default until the baseline of the
# coincidence average is fixed upstream.
FIT_WINDOW = tuple(int(x) for x in os.environ["FIT_WINDOW"].split(",")) \
    if os.environ.get("FIT_WINDOW") else (20, 300)

# deconv_<array>_run<run>_<column>.txt
COINC_NAME_RE = re.compile(r"deconv_(.+)_run(\d+)_(\w+)\.txt$")


def load_deconvolved(template, column, run="all", data_dir=COINC_DATA):
    """{(array, run): s(t)} from deconvolve/output/data/coincidence/<template>/."""
    folder = Path(data_dir) / template
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} does not exist -- "
                                f"run deconvolve/deconvolve_coincidence.py first")

    out = {}
    for path in sorted(folder.glob(f"deconv_*_{column}.txt")):
        m = COINC_NAME_RE.match(path.name)
        if not m:
            continue
        array, file_run, col = m.group(1), m.group(2), m.group(3)
        if col != column:
            continue
        if run != "all" and file_run != run:
            continue
        err_path = path.with_name(path.stem + "_err.txt")
        err = paths.load_waveform_txt(err_path) if err_path.is_file() else None
        out[(array, file_run)] = (paths.load_waveform_txt(path), err)

    if not out:
        raise FileNotFoundError(
            f"no deconv_*_{column}.txt for run '{run}' in {folder}"
            + ("  (the coincidence code does not produce the 'mode' column yet)"
               if column == "mode" else ""))
    return out


def load_legacy_deconvolved(data_dir=DECONV_DATA):
    """{(channel, ''): s(t)} from the flat deconv_<ch>.txt of deconvolve.py
    (the ROOT pipeline), kept so those results stay reachable."""
    out = {}
    for path in sorted(Path(data_dir).glob("deconv_*.txt")):
        m = re.match(r"deconv_(\d+)\.txt$", path.name)
        if m:
            out[(m.group(1), "legacy")] = (paths.load_waveform_txt(path), None)
    if not out:
        raise FileNotFoundError(f"no deconv_<ch>.txt in {data_dir}")
    return out


def _emg(t, t0, tau, sigma):
    """Decaying exponential (onset t0, constant tau) convolved with a
    Gaussian (sigma), normalized to unit area.

    Numerically stable piecewise evaluation:
      u >= 0: erfcx(u) * gauss           (erfcx of a positive is small)
      u <  0: 2*exp(arg) - erfcx(-u)*gauss, with arg < 0 guaranteed
    avoiding the overflow of erfcx(u -> -inf) in the tail."""
    x = np.asarray(t, dtype=float) - t0
    u = sigma / (np.sqrt(2) * tau) - x / (np.sqrt(2) * sigma)
    gauss = np.exp(-0.5 * (x / sigma) ** 2)

    out = np.empty_like(x)
    pos = u >= 0
    out[pos] = erfcx(u[pos]) * gauss[pos]
    neg = ~pos
    arg = sigma ** 2 / (2 * tau ** 2) - x[neg] / tau
    out[neg] = 2.0 * np.exp(arg) - erfcx(-u[neg]) * gauss[neg]
    return (0.5 / tau) * out


def _scint_model(t, N_f, tau_f, N_s, tau_s, t0, sigma, C):
    """Gabriel's model: (fast exp + slow exp) (x) Gaussian + C.
    N_f, N_s are already the integrals (p.e.) of each component."""
    return (N_f * _emg(t, t0, tau_f, sigma)
            + N_s * _emg(t, t0, tau_s, sigma) + C)


def fit_deconvolved(deconvolved, outdir_plots, outdir_data):
    """Fit the deconvolved signal with 2 exponentials (x) Gaussian + C
    (7 parameters). Parametrized by the areas N_f, N_s, so the fast/slow
    integrals come straight out of the fit.

    Returns {channel: Minuit}."""
    results = {}
    rows = []
    for (array, run), (s, s_err) in sorted(deconvolved.items()):
        tag = f"{array}_run{run}"
        t0_guess = float(np.argmax(s))
        lo, hi = FIT_WINDOW
        t = np.arange(lo, hi, dtype=float)
        y = s[lo:hi]

        # noise floor: fluctuation of the pre-signal region of the
        # deconvolved signal (what the propagated error cannot know about,
        # since 10 % of a baseline that sits at ~0 is ~0)
        pre_hi = max(int(t0_guess) - 4 * GAUSS_SIGMA_TICKS, lo + 5)
        floor = max(float(np.std(s[lo:pre_hi])), 1e-3)

        if s_err is None:
            y_err = floor * np.ones_like(y)
            err_note = f"$\\pm$ {floor:.2f} p.e./tick (pre-signal RMS)"
        else:
            # propagated 10 % of the aggregate, in quadrature with the floor
            y_err = np.sqrt(s_err[lo:hi] ** 2 + floor ** 2)
            err_note = (f"{MEAN_REL_ERR:.0%} on the average, propagated "
                        f"$\\oplus$ {floor:.2f} p.e./tick")

        model = LeastSquares(t, y, y_err, _scint_model)
        m = Minuit(model,
                   N_f=float(s.max() * GAUSS_SIGMA_TICKS * 2.5),
                   tau_f=0.3,
                   N_s=float(max(s[lo:hi].sum() / 2, 1.0)),
                   tau_s=90.0,
                   t0=t0_guess,
                   sigma=float(GAUSS_SIGMA_TICKS),
                   C=0.0)
        m.limits["N_f"]   = (0, None)
        m.limits["N_s"]   = (0, None)
        m.limits["tau_f"] = (0.01, 5)
        m.limits["tau_s"] = (10, 500)
        m.limits["t0"]    = (t0_guess - 10, t0_guess + 10)
        m.limits["sigma"] = (1, 10)
        if FIX_TAU_F_NS is not None:
            m.values["tau_f"] = FIX_TAU_F_NS / TICK_NS
            m.fixed["tau_f"] = True
        m.migrad()
        if not m.valid:                      # tau_f is barely resolved at
            m.simplex()                      # 16 ns/tick: help migrad out of
            m.migrad()                       # the flat direction
        m.hesse()
        results[(array, run)] = m

        v = m.values
        n_tot = v["N_f"] + v["N_s"]
        chi2_ndf = m.fval / (len(t) - m.nfit)

        # fast fraction f = N_f / (N_f + N_s): needs the covariance, the two
        # areas are strongly anti-correlated
        cov = m.covariance
        d_f, d_s = v["N_s"] / n_tot ** 2, -v["N_f"] / n_tot ** 2
        var_ff = (d_f ** 2 * cov["N_f", "N_f"] + d_s ** 2 * cov["N_s", "N_s"]
                  + 2 * d_f * d_s * cov["N_f", "N_s"])
        err_ff = float(np.sqrt(max(var_ff, 0.0)))

        # Birge ratio: with chi2/ndf >> 1 the Hesse errors are known to be
        # underestimated; scaling by sqrt(chi2/ndf) is the usual first aid
        birge = float(np.sqrt(chi2_ndf))
        label = (f"fit (2 exp $\\otimes$ gauss)\n"
                 f"$\\tau_f$ = {v['tau_f']*TICK_NS:.1f} ns, "
                 f"$\\tau_s$ = {v['tau_s']*TICK_NS:.0f} ns\n"
                 f"$N_f$ = {v['N_f']:.0f}, $N_s$ = {v['N_s']:.0f} p.e.\n"
                 f"fast/total = {v['N_f']/n_tot:.2f}\n"
                 f"$\\sigma$ = {v['sigma']*TICK_NS:.0f} ns, "
                 f"$\\chi^2$/ndf = {chi2_ndf:.1f}")

        y_fit = _scint_model(t, *m.values)
        t_us = t * TICK_NS / 1000.0

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(t_us, y, yerr=y_err, fmt=".", ms=3, color="k",
                    ecolor="0.55", elinewidth=0.6, capsize=0,
                    label=f"deconvolved\n{err_note}")
        ax.plot(t_us, y_fit, "-", lw=1.5, color="r", label=label)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e0)          # log in y only, from 10^0 up
        ax.set_xlim(0, 5)
        ax.set_xlabel(r"Time [$\mu$s]")
        ax.set_ylabel("p.e. / tick")
        ax.set_title(f"{array} - run {run}", fontsize="large")
        ax.legend(fontsize="small")
        fig.savefig(outdir_plots / f"fit_{tag}.png")
        plt.close(fig)

        rows.append(f"{array:>6} {run:>8} {v['tau_f']*TICK_NS:>9.4g} "
                    f"{m.errors['tau_f']*TICK_NS:>9.4g} {v['tau_s']*TICK_NS:>9.4g} "
                    f"{m.errors['tau_s']*TICK_NS:>9.4g} {v['N_f']:>9.4g} "
                    f"{m.errors['N_f']:>9.4g} {v['N_s']:>9.4g} {m.errors['N_s']:>9.4g} "
                    f"{v['N_f']/n_tot:>9.4g} {err_ff:>9.4g} "
                    f"{v['sigma']*TICK_NS:>9.4g} {m.errors['sigma']*TICK_NS:>9.4g} "
                    f"{v['C']:>9.4g} {m.errors['C']:>9.4g} "
                    f"{chi2_ndf:>9.4g} {birge:>7.3g} {int(m.valid):>5d}")

        if not m.valid:
            print(f"  !! {array} run {run}: migrad did NOT converge")
        print(f"{array} run {run}: tau_f={v['tau_f']*TICK_NS:.1f} ns, "
              f"tau_s={v['tau_s']*TICK_NS:.0f} ns, "
              f"N_f={v['N_f']:.0f}, N_s={v['N_s']:.0f}, "
              f"fast/total={v['N_f']/n_tot:.2f}, "
              f"chi2/ndf={chi2_ndf:.1f}, valid={m.valid}")

    header = (f"{'array':>6} {'run':>8} {'tau_f_ns':>9} {'err':>9} {'tau_s_ns':>9} "
              f"{'err':>9} {'N_f':>9} {'err':>9} {'N_s':>9} {'err':>9} "
              f"{'fastfrac':>9} {'err':>9} {'sigma_ns':>9} {'err':>9} "
              f"{'C':>9} {'err':>9} {'chi2_ndf':>9} {'birge':>7} {'valid':>5}")
    (outdir_data / "fit_results.txt").write_text("\n".join([header] + rows) + "\n")
    print(f"\nResults in {outdir_data / 'fit_results.txt'}")

    return results


def ask(prompt, options, given=None):
    value = (given or input(f"{prompt} {list(options)}: ")).strip().lower()
    if value not in options:
        raise SystemExit(f"invalid choice '{value}' -- expected one of {list(options)}")
    return value


def main():
    argv = sys.argv[1:]

    if argv and argv[0] == "--legacy":
        deconvolved = load_legacy_deconvolved()
        outdir_plots = paths.out(FIT_DIR, "plots", "legacy")
        outdir_data  = paths.out(FIT_DIR, "data", "legacy")
    else:
        template = ask("SPE template used in the deconvolution", TEMPLATES,
                       argv[0] if len(argv) > 0 else None)
        column   = ask("Coincidence aggregate to fit", COLUMNS,
                       argv[1] if len(argv) > 1 else None)
        run      = (argv[2] if len(argv) > 2
                    else input("Run number (e.g. 039500) or 'all': ").strip())
        deconvolved = load_deconvolved(template, column, run)
        outdir_plots = paths.out(FIT_DIR, "plots", template, column)
        outdir_data  = paths.out(FIT_DIR, "data", template, column)
        print(f"\ntemplate={template}  column={column}  run={run}  "
              f"-> {len(deconvolved)} waveform(s)\n")

    return fit_deconvolved(deconvolved, outdir_plots, outdir_data)


if __name__ == "__main__":
    main()
