"""Fit of the deconvolved signal: 2 exponentials (x) Gaussian + C.

Analysis step 3: reads the deconv_<ch>.txt files produced by deconvolve/
and fits the LAr scintillation model.

Usage:
    python fit/fit.py           (or: cd fit && python fit.py)

Outputs under fit/output/:
    plots/fit_<ch>.png    data + fit (linear and log)
    data/fit_results.txt  table with the parameters of every channel
"""
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
from deconvolve.config import GAUSS_SIGMA_TICKS, TICK_NS

FIT_DIR = paths.FIT_DIR
DECONV_DATA = paths.DECONV_DIR / "output" / "data"

# fit window on the deconvolved signal (ticks): avoids the FFT window edges
# and the template wrap region (~845)
FIT_WINDOW = (20, 700)


def load_deconvolved(data_dir=DECONV_DATA):
    """{channel: s(t)} from the deconv_<ch>.txt files of deconvolve/."""
    out = {}
    for path in sorted(Path(data_dir).glob("deconv_*.txt")):
        m = re.match(r"deconv_(\d+)\.txt$", path.name)
        if m:
            out[int(m.group(1))] = paths.load_waveform_txt(path)
    if not out:
        raise FileNotFoundError(
            f"no deconv_<ch>.txt in {data_dir} -- run deconvolve/deconvolve.py first")
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


def fit_deconvolved(deconvolved):
    """Fit the deconvolved signal with 2 exponentials (x) Gaussian + C
    (7 parameters). Parametrized by the areas N_f, N_s, so the fast/slow
    integrals come straight out of the fit.

    Returns {channel: Minuit}."""
    outdir   = paths.out(FIT_DIR, "plots")
    data_dir = paths.out(FIT_DIR, "data")

    results = {}
    rows = []
    for ch, s in deconvolved.items():
        t0_guess = float(np.argmax(s))
        lo, hi = FIT_WINDOW
        t = np.arange(lo, hi, dtype=float)
        y = s[lo:hi]

        # per-tick error: fluctuation of the pre-signal region of the deconvolved signal
        pre_hi = max(int(t0_guess) - 4 * GAUSS_SIGMA_TICKS, lo + 5)
        y_err = max(float(np.std(s[lo:pre_hi])), 1e-3) * np.ones_like(y)

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
        m.migrad()
        m.hesse()
        results[ch] = m

        v = m.values
        n_tot = v["N_f"] + v["N_s"]
        chi2_ndf = m.fval / (len(t) - m.nfit)
        label = (f"fit (2 exp $\\otimes$ gauss)\n"
                 f"$\\tau_f$ = {v['tau_f']*TICK_NS:.1f} ns, "
                 f"$\\tau_s$ = {v['tau_s']*TICK_NS:.0f} ns\n"
                 f"$N_f$ = {v['N_f']:.0f}, $N_s$ = {v['N_s']:.0f} p.e.\n"
                 f"fast/total = {v['N_f']/n_tot:.2f}\n"
                 f"$\\sigma$ = {v['sigma']*TICK_NS:.0f} ns, "
                 f"$\\chi^2$/ndf = {chi2_ndf:.1f}")

        y_fit = _scint_model(t, *m.values)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        ax1.plot(t, y, ".", ms=2, color="k", label="deconvolved")
        ax1.plot(t, y_fit, "-", color="r", label=label)
        ax1.set_ylabel("p.e. / tick")
        ax1.set_title(f"Ch {ch} - deconvolved + fit")
        ax1.legend(fontsize=8)
        ax2.plot(t, y, ".", ms=2, color="k")
        ax2.plot(t, y_fit, "-", color="r")
        ax2.set_yscale("log")
        ax2.set_ylim(bottom=1e0)
        ax2.set_xlim(0, 500)
        ax2.set_ylabel("p.e. / tick (log)")
        ax2.set_xlabel("Sample")
        plt.tight_layout()
        plt.savefig(outdir / f"fit_{ch}.png")
        plt.close(fig)

        rows.append((ch, v['tau_f']*TICK_NS, m.errors['tau_f']*TICK_NS,
                     v['tau_s']*TICK_NS, m.errors['tau_s']*TICK_NS,
                     v['N_f'], m.errors['N_f'], v['N_s'], m.errors['N_s'],
                     v['N_f']/n_tot, v['sigma']*TICK_NS, v['C'],
                     chi2_ndf, int(m.valid)))

        print(f"Channel {ch}: tau_f={v['tau_f']*TICK_NS:.1f} ns, "
              f"tau_s={v['tau_s']*TICK_NS:.0f} ns, "
              f"N_f={v['N_f']:.0f}, N_s={v['N_s']:.0f}, "
              f"fast/total={v['N_f']/n_tot:.2f}, "
              f"chi2/ndf={chi2_ndf:.1f}, valid={m.valid}")

    header = ("channel tau_f_ns err_tau_f tau_s_ns err_tau_s N_f err_N_f "
              "N_s err_N_s fast_frac sigma_ns C chi2_ndf valid")
    np.savetxt(data_dir / "fit_results.txt", np.array(rows),
               fmt="%.6g", header=header)
    print(f"\nResults in {data_dir / 'fit_results.txt'}")

    return results


def main():
    return fit_deconvolved(load_deconvolved())


if __name__ == "__main__":
    main()
