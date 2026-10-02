#!/usr/bin/env python3
"""
toy_MC.py -- Toy Monte Carlo validation of the forward-fold fit
(fitting-bertolini approach) with white Gaussian noise.

Procedure (per truth point):
  1. Build the expected waveform mu(t) = (s_true * h)(t), where s_true is the
     double-exponential LAr emission (bin-integrated) and h is the SPE template.
  2. Add white Gaussian noise with sigma = NOISE_SIGMA.
  3. Estimate yerr from the noise region (t > NOISE_REGION_T), as in the data fit.
  4. Fit with the same model / fixed parameters / limits used on data.
  5. Repeat N_TOYS times and study bias, pulls and chi2.

IMPORTANT: the toy is only meaningful if MODEL below is *identical* to the one in
fitting-bertolini.py (template alignment, normalization, parametrization).
If yours differs, replace `emission()` / `model()` with your own functions and
fill TRUTH_POINTS with parameters obtained from that same model.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy import stats

try:
    import dunestyle.matplotlib as dunestyle   # applies the DUNE rcParams on import
    HAS_DUNESTYLE = True
except ImportError:
    dunestyle = None
    HAS_DUNESTYLE = False
    print("[toy MC] dunestyle not found -- using matplotlib defaults")

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# SPE template (Francesca, channel 2081).
REPO_DIR = SCRIPT_DIR.parent   # fitting -> sag_analysis
TEMPLATE_PATH = (REPO_DIR / "calib" / "output" / "data" / "templates"
                 / "francesca_template" / "template_42156_M8_2.txt")

DT = 0.016            # sampling period [us]
T_MAX = 10.0          # length of the simulated window [us]
NOISE_REGION_T = 8.0  # yerr is the std of the waveform for t > this value [us]

# White noise level of the *averaged* waveform [ADC]. Use the same value your
# data fit gets from the noise region (std for t > 8 us).
NOISE_SIGMA = 1.0

TAU_FAST = 0.006      # fixed singlet lifetime [us]
T0_CENTER = 1.008     # [us]
T0_HALF_WIDTH = 0.010 # t0 limits: centre +/- T0_HALF_WIDTH [us]
# In the data the t0 window is centred on the mean t0. To mimic that, each toy fit
# centres its t0 limits on the true t0 of the point (True), instead of on the fixed
# T0_CENTER (False). With False, grid points far from T0_CENTER sit next to a limit.
T0_LIMITS_ON_TRUTH = True

N_TOYS = 1000
RNG_SEED = 12345

# Initial guesses for each toy fit:
#   'truth'     -> start exactly at the true values (optimistic)
#   'perturbed' -> truth * (1 + SEED_SMEAR * N(0,1)), t0 kept inside limits
SEED_MODE = "perturbed"
SEED_SMEAR = 0.20
MAX_RETRIES = 2       # extra simplex+migrad passes when a fit is not valid

# Truth points: fill with the fit results from fitting-bertolini (one per field).
# A_* are amplitudes (integral of each component = A * tau), same units as your model.
# EXAMPLE VALUES -- replace with your own.
TRUTH_POINTS = [
    dict(label="E0",     field_kV=0.0,   A_fast=4200.0, A_slow=52.0, tau_slow=1.45, t0=1.008),
    dict(label="E77p8",  field_kV=77.8,  A_fast=3800.0, A_slow=50.0, tau_slow=1.40, t0=1.008),
    dict(label="E155p7", field_kV=155.7, A_fast=3500.0, A_slow=48.0, tau_slow=1.35, t0=1.008),
]

# -----------------------------------------------------------------------------
# SCAN MODE
#   'fields' -> use TRUTH_POINTS above (one point per field)
#   'grid'   -> scan a grid in (tau_slow, F_prompt, t0, noise level) around a
#               base point; the total light I_total is kept fixed
# -----------------------------------------------------------------------------
SCAN_MODE = "grid"

GRID_I_TOTAL = None                    # None -> take I_fast+I_slow from TRUTH_POINTS[0]
GRID_TAU_SLOW = [1.20, 1.35, 1.50, 1.65, 1.80]   # [us]
GRID_F_PROMPT = [0.15, 0.25, 0.35]
# t0 relative to the sampling grid: 1.008 = bin edge (63*16 ns), 1.012 = quarter,
# 1.016 = bin centre.
GRID_T0 = [1.008, 1.012, 1.016]
GRID_NOISE_FACTOR = [0.5, 1.0, 2.0]    # multiplies NOISE_SIGMA

# Per-point plots (5 PNGs per point). With a big grid, set False and use the
# grid summary plots + CSV instead.
MAKE_POINT_PLOTS = False

FREE_PARAMS = ["A_fast", "A_slow", "tau_slow", "t0"]
DERIVED = ["I_fast", "I_slow", "F_prompt", "I_dif"]

OUTPUT_DIR = SCRIPT_DIR / "output" / "toy_mc"


# =============================================================================
# TEMPLATE AND MODEL
# =============================================================================
def load_template(path):
    """Load the SPE template; if the file has several columns, use the last one."""
    data = np.loadtxt(path)
    if data.ndim == 2:
        data = data[:, -1]
    return np.asarray(data, dtype=float)


def emission(t, A, tau, t0):
    """
    Bin-integrated exponential emission, in counts per sample.
    Integrating analytically over [t_k, t_k + DT] keeps the integral A*tau exact
    even when tau << DT (fast component) and makes the model smooth in t0.
    """
    a = np.clip(t - t0, 0.0, None)
    b = np.clip(t + DT - t0, 0.0, None)
    return A * tau * (np.exp(-a / tau) - np.exp(-b / tau))


def make_model(template):
    """Return model(t, A_fast, A_slow, tau_slow, t0, tau_fast) = (s * h)(t)."""
    def model(t, A_fast, A_slow, tau_slow, t0, tau_fast):
        s = emission(t, A_fast, tau_fast, t0) + emission(t, A_slow, tau_slow, t0)
        return np.convolve(s, template)[: len(t)]
    return model


# =============================================================================
# DERIVED QUANTITIES AND ERROR PROPAGATION
# =============================================================================
def derived_values(p):
    """p = (A_fast, A_slow, tau_slow, t0) -> (I_fast, I_slow, F_prompt, I_dif)."""
    A_f, A_s, tau_s, _ = p
    I_f = A_f * TAU_FAST
    I_s = A_s * tau_s
    return np.array([I_f, I_s, I_f / (I_f + I_s), I_f - I_s])


def derived_with_errors(p, cov):
    """Linear error propagation with a central-difference Jacobian."""
    p = np.asarray(p, dtype=float)
    f0 = derived_values(p)
    J = np.zeros((len(f0), len(p)))
    for j in range(len(p)):
        h = 1e-6 * max(abs(p[j]), 1e-6)
        dp = np.zeros_like(p)
        dp[j] = h
        J[:, j] = (derived_values(p + dp) - derived_values(p - dp)) / (2 * h)
    var = np.einsum("ij,jk,ik->i", J, cov, J)
    return f0, np.sqrt(np.clip(var, 0.0, None))


# =============================================================================
# SINGLE FIT
# =============================================================================
def initial_guess(truth, rng):
    if SEED_MODE == "truth":
        return {k: truth[k] for k in FREE_PARAMS}
    guess = {}
    for k in ["A_fast", "A_slow", "tau_slow"]:
        guess[k] = abs(truth[k] * (1.0 + SEED_SMEAR * rng.standard_normal()))
    guess["t0"] = np.clip(truth["t0"] + 0.5 * T0_HALF_WIDTH * rng.standard_normal(),
                          t0_centre(truth) - T0_HALF_WIDTH, t0_centre(truth) + T0_HALF_WIDTH)
    return guess


def t0_centre(truth):
    return truth["t0"] if T0_LIMITS_ON_TRUTH else T0_CENTER


def params_at_limit(m, rtol=1e-4):
    """True if any free parameter sits on (or numerically at) one of its limits."""
    for p in m.params:
        if p.is_fixed:
            continue
        lo, hi = p.lower_limit, p.upper_limit
        span = (hi - lo) if (lo is not None and hi is not None) else max(abs(p.value), 1.0)
        if lo is not None and p.value - lo <= rtol * span:
            return True
        if hi is not None and hi - p.value <= rtol * span:
            return True
    return False


def fit_waveform(t, y, model, guess, t0c=T0_CENTER):
    """Fit exactly as on data: LeastSquares, yerr from noise region, tau_fast fixed."""
    sigma = np.std(y[t > NOISE_REGION_T], ddof=1)
    yerr = np.full_like(y, sigma)

    cost = LeastSquares(t, y, yerr, model)
    m = Minuit(cost, tau_fast=TAU_FAST, **guess)
    m.fixed["tau_fast"] = True
    m.limits["A_fast"] = (0.0, None)
    m.limits["A_slow"] = (0.0, None)
    m.limits["tau_slow"] = (0.1, 5.0)
    m.limits["t0"] = (t0c - T0_HALF_WIDTH, t0c + T0_HALF_WIDTH)
    m.migrad()
    # Retry a couple of times if the EDM goal was not reached: the fast component
    # (tau_fast << DT) makes the chi2 surface sharp in t0 near bin edges.
    for _ in range(MAX_RETRIES):
        if m.valid:
            break
        m.simplex().migrad()
    m.hesse()
    return m, sigma


# =============================================================================
# TOY LOOP
# =============================================================================
def run_toys(truth, t, model, n_toys, rng):
    true_p = np.array([truth[k] for k in FREE_PARAMS])
    true_d = derived_values(true_p)
    mu = model(t, *true_p, TAU_FAST)

    est = np.full((n_toys, len(FREE_PARAMS)), np.nan)
    err = np.full_like(est, np.nan)
    d_est = np.full((n_toys, len(DERIVED)), np.nan)
    d_err = np.full_like(d_est, np.nan)
    chi2 = np.full(n_toys, np.nan)
    ndof = np.full(n_toys, np.nan)
    valid = np.zeros(n_toys, dtype=bool)
    limit = np.zeros(n_toys, dtype=bool)
    example = None

    noise_sigma = truth.get("noise_sigma", NOISE_SIGMA)
    for i in range(n_toys):
        y = mu + rng.normal(0.0, noise_sigma, size=t.size)
        m, sigma = fit_waveform(t, y, model, initial_guess(truth, rng), t0_centre(truth))

        # a fit parked on a limit has meaningless Hesse errors -> excluded from the
        # statistics and counted as failed (fraction reported separately as at_limit)
        limit[i] = m.fmin.has_parameters_at_limit or params_at_limit(m)
        valid[i] = m.valid and m.accurate and not limit[i]
        if not valid[i]:
            continue

        p = np.array([m.values[k] for k in FREE_PARAMS])
        cov = np.array(m.covariance)
        idx = [m.parameters.index(k) for k in FREE_PARAMS]
        cov = cov[np.ix_(idx, idx)]

        est[i] = p
        err[i] = [m.errors[k] for k in FREE_PARAMS]
        d_est[i], d_err[i] = derived_with_errors(p, cov)
        chi2[i] = m.fval
        ndof[i] = m.ndof
        if example is None:
            example = dict(y=y, fit=model(t, *p, TAU_FAST), sigma=sigma)

    return dict(truth=truth, true_p=true_p, true_d=true_d, mu=mu,
                est=est, err=err, d_est=d_est, d_err=d_err,
                chi2=chi2, ndof=ndof, valid=valid, limit=limit, example=example)


def summarize(res):
    """Bias and pull statistics for free and derived parameters."""
    ok = res["valid"]
    names = FREE_PARAMS + DERIVED
    est = np.hstack([res["est"], res["d_est"]])[ok]
    err = np.hstack([res["err"], res["d_err"]])[ok]
    true = np.concatenate([res["true_p"], res["true_d"]])
    n = ok.sum()

    rows = {}
    for j, name in enumerate(names):
        delta = est[:, j] - true[j]
        pull = delta / err[:, j]
        pull = pull[np.isfinite(pull)]
        rows[name] = dict(
            true=true[j],
            mean_est=np.mean(est[:, j]),
            bias=np.mean(delta),
            bias_err=np.std(delta, ddof=1) / np.sqrt(n),
            rel_bias=np.mean(delta) / true[j] if true[j] != 0 else np.nan,
            spread=np.std(est[:, j], ddof=1),
            mean_err=np.mean(err[:, j]),
            pull_mean=np.mean(pull),
            pull_mean_err=np.std(pull, ddof=1) / np.sqrt(pull.size),
            pull_width=np.std(pull, ddof=1),
            pull_width_err=np.std(pull, ddof=1) / np.sqrt(2 * (pull.size - 1)),
        )
    return rows


def write_summary(results, summaries, path):
    with open(path, "w") as f:
        f.write("# Toy MC summary -- white noise\n")
        f.write(f"# N_TOYS={N_TOYS}  NOISE_SIGMA={NOISE_SIGMA}  DT={DT} us  "
                f"TAU_FAST={TAU_FAST} us (fixed)  SEED_MODE={SEED_MODE}\n")
        f.write(f"# template: {TEMPLATE_PATH.name}\n")
        for res, rows in zip(results, summaries):
            tr = res["truth"]
            fail = 1.0 - res["valid"].mean()
            lim = res["limit"].mean()
            f.write(f"\n# point {tr['label']}  field={tr['field_kV']} kV  "
                    f"failure_rate={fail:.4f}  at_limit_rate={lim:.4f}\n")
            f.write(f"{'param':<10}{'true':>14}{'bias':>14}{'bias_err':>12}{'rel_bias':>11}"
                    f"{'spread':>12}{'mean_err':>12}{'pull_mu':>9}{'+-':>7}"
                    f"{'pull_sig':>9}{'+-':>7}\n")
            for name, r in rows.items():
                f.write(f"{name:<10}{r['true']:>14.6g}{r['bias']:>14.4g}{r['bias_err']:>12.3g}"
                        f"{r['rel_bias']:>11.2e}{r['spread']:>12.4g}{r['mean_err']:>12.4g}"
                        f"{r['pull_mean']:>9.3f}{r['pull_mean_err']:>7.3f}"
                        f"{r['pull_width']:>9.3f}{r['pull_width_err']:>7.3f}\n")


# =============================================================================
# PLOT STYLE
# =============================================================================
LABELS = {
    "A_fast": r"$A_{\rm fast}$", "A_slow": r"$A_{\rm slow}$",
    "tau_slow": r"$\tau_{\rm slow}$", "t0": r"$t_0$",
    "I_fast": r"$I_{\rm fast}$", "I_slow": r"$I_{\rm slow}$",
    "F_prompt": r"$F_{\rm prompt}$", "I_dif": r"$I_{\rm fast}-I_{\rm slow}$",
}

# Okabe-Ito, same order as the dunestyle color cycler
OI = ["#000000", "#D55E00", "#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#0072B2"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# Smaller fonts for the multi-panel diagnostic plots (dunestyle defaults are
# sized for single-panel figures).
DIAG_RC = {
    "axes.labelsize": 12, "axes.titlesize": 12, "figure.titlesize": 14,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
}

PAPER_FORMATS = ("png", "pdf")


def sim_label(ax, fontsize=None, **kwargs):
    """'DUNE Simulation' watermark (plain text fallback without dunestyle)."""
    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    if HAS_DUNESTYLE:
        dunestyle.Simulation(ax=ax, **kwargs)
    else:
        ax.text(kwargs.pop("x", 0.05), kwargs.pop("y", 0.90), "Simulation",
                transform=ax.transAxes, fontsize=kwargs.get("fontsize", 16), fontweight="bold")


def save(fig, path, formats=("png",)):
    path = Path(path)
    for ext in formats:
        fig.savefig(path.with_suffix("." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)


def all_arrays(res):
    """Stacked (free + derived) estimates, errors and truths for the valid toys."""
    ok = res["valid"]
    est = np.hstack([res["est"], res["d_est"]])[ok]
    err = np.hstack([res["err"], res["d_err"]])[ok]
    true = np.concatenate([res["true_p"], res["true_d"]])
    return est, err, true


def pulls_of(res, name):
    est, err, true = all_arrays(res)
    j = (FREE_PARAMS + DERIVED).index(name)
    p = (est[:, j] - true[j]) / err[:, j]
    return p[np.isfinite(p)]


def pull_stats(p):
    n = p.size
    w = np.std(p, ddof=1)
    return np.mean(p), w / np.sqrt(n), w, w / np.sqrt(2 * (n - 1))


def true_F(res):
    return res["true_d"][DERIVED.index("F_prompt")]


def noise_factor(res):
    return res["truth"].get("noise_factor", 1.0)


def bin_phase_ns(t0):
    """Position of t0 inside its sample, in ns from the lower sample edge."""
    x = t0 / DT
    return (x - np.floor(x + 1e-9)) * DT * 1e3


# =============================================================================
# DIAGNOSTIC PLOTS (per point)
# =============================================================================
def plot_example(res, t, path):
    ex = res["example"]
    if ex is None:
        return
    with plt.rc_context(DIAG_RC):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                     gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))
        a1.plot(t, ex["y"], color="0.65", lw=0.8, label="toy waveform")
        a1.plot(t, res["mu"], color=OI[0], lw=1.8, label="truth")
        a1.plot(t, ex["fit"], color=OI[1], lw=1.4, ls="--", label="fit")
        a1.set_ylabel("Amplitude [ADC]")
        a1.legend(loc="upper right")
        sim_label(a1, fontsize=12)
        a1.text(0.05, 0.80, res["truth"]["label"], transform=a1.transAxes, fontsize=10)
        a2.plot(t, (ex["y"] - ex["fit"]) / ex["sigma"], color="0.3", lw=0.6)
        a2.axhline(0, color="k", lw=0.8)
        a2.set_ylabel(r"(toy$-$fit)/$\sigma$")
        a2.set_xlabel(r"Time [$\mu$s]")
        save(fig, path)


def plot_pulls(res, rows, path):
    names = FREE_PARAMS + DERIVED
    x = np.linspace(-5, 5, 300)
    with plt.rc_context(DIAG_RC):
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        for ax, name in zip(axes.flat, names):
            p = pulls_of(res, name)
            r = rows[name]
            ax.hist(p, bins=40, range=(-5, 5), density=True, histtype="stepfilled",
                    color=OI[2], alpha=0.35)
            ax.hist(p, bins=40, range=(-5, 5), density=True, histtype="step", color=OI[6])
            ax.plot(x, stats.norm.pdf(x), "k--", lw=1.2, label=r"$\mathcal{N}(0,1)$")
            ax.plot(x, stats.norm.pdf(x, r["pull_mean"], r["pull_width"]), color=OI[1], lw=1.5,
                    label=rf"$\mu$={r['pull_mean']:.2f}$\pm${r['pull_mean_err']:.2f}" "\n"
                          rf"$\sigma$={r['pull_width']:.2f}$\pm${r['pull_width_err']:.2f}")
            ax.set_xlabel(f"pull {LABELS[name]}")
            ax.set_ylim(0, 0.62)
            ax.legend(loc="upper right")
        sim_label(axes[0, 0], fontsize=11)
        fig.suptitle(f"Pulls -- {res['truth']['label']}  ({res['valid'].sum()} valid toys)")
        fig.tight_layout()
        save(fig, path)


def plot_rel_bias_hist(res, rows, path):
    est, _, true = all_arrays(res)
    names = FREE_PARAMS + DERIVED
    with plt.rc_context(DIAG_RC):
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        for j, (ax, name) in enumerate(zip(axes.flat, names)):
            if name == "t0":
                dev = (est[:, j] - true[j]) * 1e3
                xlabel = r"$\hat t_0 - t_{0}^{\rm true}$ [ns]"
            else:
                dev = (est[:, j] - true[j]) / abs(true[j]) * 100
                xlabel = f"{LABELS[name]}: " + r"$(\hat\theta-\theta^{\rm true})/\theta^{\rm true}$ [%]"
            ax.hist(dev, bins=40, histtype="step", color=OI[4], lw=1.5)
            ax.axvline(0, color="k", lw=1)
            ax.axvline(np.mean(dev), color=OI[1], ls="--", lw=1.5, label=f"mean = {np.mean(dev):.3g}")
            ax.set_xlabel(xlabel)
            ax.legend(loc="upper right")
        sim_label(axes[0, 0], fontsize=11)
        fig.suptitle(f"Deviation from truth -- {res['truth']['label']}")
        fig.tight_layout()
        save(fig, path)


def plot_chi2(res, path):
    ok = res["valid"]
    chi2 = res["chi2"][ok]
    ndof = int(np.median(res["ndof"][ok]))
    with plt.rc_context(DIAG_RC):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(chi2, bins=40, density=True, histtype="step", color=OI[6], lw=1.5, label="toys")
        x = np.linspace(chi2.min(), chi2.max(), 300)
        ax.plot(x, stats.chi2.pdf(x, ndof), "k--", lw=1.4, label=rf"$\chi^2$ pdf, ndof = {ndof}")
        ax.set_xlabel(r"$\chi^2$")
        ax.set_ylabel("Probability density")
        ax.set_ylim(0, 1.3 * ax.get_ylim()[1])
        ax.legend(loc="upper right", title=rf"$\langle\chi^2\rangle$/ndof = {np.mean(chi2) / ndof:.3f}")
        sim_label(ax, fontsize=12)
        save(fig, path)


def plot_correlation(res, path):
    corr = np.corrcoef(res["est"][res["valid"]].T)
    ticks = [LABELS[k] for k in FREE_PARAMS]
    with plt.rc_context(DIAG_RC):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(ticks)), ticks)
        ax.set_yticks(range(len(ticks)), ticks)
        ax.minorticks_off()
        ax.tick_params(top=False, right=False)
        for i in range(len(ticks)):
            for j in range(len(ticks)):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if abs(corr[i, j]) > 0.6 else "black")
        fig.colorbar(im, ax=ax, label=r"$\rho$ (toy-to-toy)")
        ax.set_title(res["truth"]["label"], fontsize=10)
        save(fig, path)


def plot_summary_vs_field(results, summaries, path):
    fields = np.array([r["truth"]["field_kV"] for r in results])
    names = FREE_PARAMS + DERIVED
    with plt.rc_context(DIAG_RC):
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        for k, name in enumerate(names):
            kw = dict(marker=MARKERS[k], ms=6, capsize=3, lw=1, color=OI[k % len(OI)],
                      label=LABELS[name])
            pm = [s[name]["pull_mean"] for s in summaries]
            pme = [s[name]["pull_mean_err"] for s in summaries]
            pw = [s[name]["pull_width"] for s in summaries]
            pwe = [s[name]["pull_width_err"] for s in summaries]
            rb = [s[name]["rel_bias"] * 100 if name != "t0" else np.nan for s in summaries]
            rbe = [s[name]["bias_err"] / abs(s[name]["true"]) * 100 if name != "t0" else np.nan
                   for s in summaries]
            axes[0].errorbar(fields, pm, pme, **kw)
            axes[1].errorbar(fields, pw, pwe, **kw)
            axes[2].errorbar(fields, rb, rbe, **kw)
        for ax, ref, lab in zip(axes, [0, 1, 0], ["Pull mean", "Pull width", "Relative bias [%]"]):
            ax.axhline(ref, color="k", lw=0.8)
            ax.set_ylabel(lab)
            ax.set_xlabel("Drift HV [kV]")
        sim_label(axes[0], fontsize=11)
        axes[2].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        save(fig, path)


# =============================================================================
# GRID SCAN
# =============================================================================
def build_grid():
    base = TRUTH_POINTS[0]
    if GRID_I_TOTAL is None:
        i_total = base["A_fast"] * TAU_FAST + base["A_slow"] * base["tau_slow"]
    else:
        i_total = GRID_I_TOTAL
    points = []
    for nf in GRID_NOISE_FACTOR:
        for t0 in GRID_T0:
            for fp in GRID_F_PROMPT:
                for ts in GRID_TAU_SLOW:
                    I_f = fp * i_total
                    I_s = (1.0 - fp) * i_total
                    points.append(dict(
                        label=f"tau{ts:.2f}_fp{fp:.2f}_t0{t0 * 1e3:.0f}ns_nf{nf:.1f}",
                        field_kV=np.nan,
                        A_fast=I_f / TAU_FAST, A_slow=I_s / ts, tau_slow=ts, t0=t0,
                        noise_sigma=NOISE_SIGMA * nf,
                        F_prompt_true=fp, noise_factor=nf))
    return points


CSV_COLS = ["label", "tau_slow_true", "F_prompt_true", "t0_true", "noise_factor",
            "failure_rate", "at_limit_rate", "param", "true", "bias", "bias_err",
            "rel_bias", "spread", "mean_err", "pull_mean", "pull_mean_err",
            "pull_width", "pull_width_err"]


def append_csv(res, rows, path):
    """Long-format table (one line per point x parameter); appended per point so a
    partial run is still usable."""
    new = not path.exists()
    tr = res["truth"]
    with open(path, "a") as f:
        if new:
            f.write(",".join(CSV_COLS) + "\n")
        for name, r in rows.items():
            vals = [tr["label"], tr["tau_slow"], tr.get("F_prompt_true", np.nan), tr["t0"],
                    tr.get("noise_factor", 1.0), 1.0 - res["valid"].mean(),
                    res["limit"].mean(), name, r["true"], r["bias"], r["bias_err"],
                    r["rel_bias"], r["spread"], r["mean_err"], r["pull_mean"],
                    r["pull_mean_err"], r["pull_width"], r["pull_width_err"]]
            f.write(",".join(str(v) for v in vals) + "\n")


def plot_grid(results, summaries, plots_dir):
    """Diagnostic overview of the grid. For each key parameter: rows = t0 position,
    cols = noise factor; each panel shows a statistic vs tau_slow, one curve per F_prompt."""
    params = ["tau_slow", "F_prompt", "I_fast", "I_slow"]
    t0s = sorted({r["truth"]["t0"] for r in results})
    nfs = sorted({noise_factor(r) for r in results})
    fps = sorted({r["truth"]["F_prompt_true"] for r in results})

    with plt.rc_context(DIAG_RC):
        for name in params:
            for stat, ref, ylab in [("pull_mean", 0.0, "Pull mean"),
                                    ("pull_width", 1.0, "Pull width"),
                                    ("rel_bias", 0.0, "Relative bias [%]")]:
                fig, axes = plt.subplots(len(t0s), len(nfs),
                                         figsize=(4.5 * len(nfs), 3.3 * len(t0s)),
                                         sharex=True, sharey=True, squeeze=False)
                for i, t0 in enumerate(t0s):
                    for j, nf in enumerate(nfs):
                        ax = axes[i, j]
                        for k, fp in enumerate(fps):
                            sel = sorted([(r["truth"]["tau_slow"], s[name])
                                          for r, s in zip(results, summaries)
                                          if r["truth"]["t0"] == t0 and noise_factor(r) == nf
                                          and r["truth"]["F_prompt_true"] == fp],
                                         key=lambda x: x[0])
                            if not sel:
                                continue
                            x = np.array([a for a, _ in sel])
                            if stat == "rel_bias":
                                y = np.array([b["rel_bias"] * 100 for _, b in sel])
                                e = np.array([b["bias_err"] / abs(b["true"]) * 100 for _, b in sel])
                            else:
                                y = np.array([b[stat] for _, b in sel])
                                e = np.array([b[stat + "_err"] for _, b in sel])
                            ax.errorbar(x, y, e, marker=MARKERS[k], ms=4, capsize=2, lw=1,
                                        color=OI[(k + 1) % len(OI)],
                                        label=rf"$F_{{\rm prompt}}$ = {fp:.2f}")
                        ax.axhline(ref, color="k", lw=0.8)
                        ax.text(0.04, 0.88, rf"$t_0$ phase {bin_phase_ns(t0):.0f} ns, noise $\times${nf:g}",
                                transform=ax.transAxes, fontsize=9)
                        if i == len(t0s) - 1:
                            ax.set_xlabel(r"$\tau_{\rm slow}^{\rm true}$ [$\mu$s]")
                        if j == 0:
                            ax.set_ylabel(ylab)
                axes[0, -1].legend(loc="lower right")
                fig.suptitle(f"{LABELS[name]}: {ylab.lower()}")
                sim_label(axes[0, 0], fontsize=10, y=1.05)
                fig.tight_layout()
                save(fig, plots_dir / f"grid_{name}_{stat}")

        fig, axes = plt.subplots(1, len(nfs), figsize=(4.5 * len(nfs), 3.8), sharey=True, squeeze=False)
        for j, nf in enumerate(nfs):
            ax = axes[0, j]
            for i, t0 in enumerate(t0s):
                sel = [(r["truth"]["tau_slow"], 1.0 - r["valid"].mean()) for r in results
                       if r["truth"]["t0"] == t0 and noise_factor(r) == nf]
                taus = sorted({a for a, _ in sel})
                y = [np.mean([b for a, b in sel if a == tau]) * 100 for tau in taus]
                ax.plot(taus, y, marker=MARKERS[i], color=OI[(i + 1) % len(OI)],
                        label=rf"$t_0$ phase {bin_phase_ns(t0):.0f} ns")
            ax.set_title(rf"noise $\times${nf:g}")
            ax.set_xlabel(r"$\tau_{\rm slow}^{\rm true}$ [$\mu$s]")
        axes[0, 0].set_ylabel("Failed fits [%]")
        axes[0, -1].legend(loc="upper right")
        fig.tight_layout()
        save(fig, plots_dir / "grid_failure_rate")


# =============================================================================
# PAPER FIGURES
# =============================================================================
def nominal_index(results, t0_target=None):
    """Grid point closest to the data configuration (noise x1, T0 of TRUTH_POINTS[0],
    and its F_prompt and tau_slow)."""
    if SCAN_MODE != "grid":
        return 0
    base = TRUTH_POINTS[0]
    f_base = base["A_fast"] * TAU_FAST / (base["A_fast"] * TAU_FAST + base["A_slow"] * base["tau_slow"])
    t0_target = base["t0"] if t0_target is None else t0_target

    def score(i):
        r = results[i]
        return (abs(noise_factor(r) - 1.0), abs(r["truth"]["t0"] - t0_target),
                abs(true_F(r) - f_base), abs(r["truth"]["tau_slow"] - base["tau_slow"]))
    return min(range(len(results)), key=score)


def centre_t0(results):
    t0s = sorted({r["truth"]["t0"] for r in results})
    return min(t0s, key=lambda t0: abs(bin_phase_ns(t0) - DT * 1e3 / 2))


def paper_example(res, t, path):
    """Toy waveform, truth and fit with normalized residuals."""
    ex = res["example"]
    if ex is None:
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))
    a1.plot(t, ex["y"], color="0.6", lw=0.8, label="Toy waveform")
    a1.plot(t, res["mu"], color=OI[0], lw=2, label="Truth")
    a1.plot(t, ex["fit"], color=OI[1], lw=1.5, ls="--", label="Fit")
    a1.set_ylabel("Amplitude [ADC]")
    a1.set_xlim(0, min(T_MAX, 8.0))
    a1.set_ylim(top=1.25 * a1.get_ylim()[1])
    a1.legend(loc="upper right", fontsize=14)
    sim_label(a1)
    r = (ex["y"] - ex["fit"]) / ex["sigma"]
    a2.plot(t, r, color="0.3", lw=0.6)
    a2.axhline(0, color="k", lw=1)
    a2.set_ylim(-4.5, 4.5)
    a2.set_ylabel(r"Pull")
    a2.set_xlabel(r"Time [$\mu$s]")
    save(fig, path, PAPER_FORMATS)


def paper_pull_summary(results, path):
    """Pull mean (marker) and pull width (bar) for every parameter at the nominal point;
    in grid mode, also for t0 at the centre of the sample, to show the bin-edge effect."""
    names = FREE_PARAMS + DERIVED
    series = [(nominal_index(results), rf"$t_0$ phase {bin_phase_ns(results[nominal_index(results)]['truth']['t0']):.0f} ns (data)")]
    if SCAN_MODE == "grid":
        tc = centre_t0(results)
        ic = nominal_index(results, t0_target=tc)
        if ic != series[0][0]:
            series.append((ic, rf"$t_0$ phase {bin_phase_ns(tc):.0f} ns (sample centre)"))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.axvspan(-1, 1, color="0.9", zorder=0)
    ax.axvline(0, color="k", lw=1, zorder=1)
    y = np.arange(len(names))[::-1]
    off = np.linspace(-0.15, 0.15, len(series)) if len(series) > 1 else [0.0]
    for k, ((idx, lab), dy) in enumerate(zip(series, off)):
        m = np.array([pull_stats(pulls_of(results[idx], n))[0] for n in names])
        w = np.array([pull_stats(pulls_of(results[idx], n))[2] for n in names])
        ax.errorbar(m, y + dy, xerr=w, fmt=MARKERS[k], color=OI[k + 1], ms=8,
                    capsize=4, lw=2, label=lab, zorder=3)
    ax.set_yticks(y, [LABELS[n] for n in names])
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.8, len(names) + 1.2)
    ax.set_xlabel("Pull mean  (bars: pull width)")
    ax.legend(loc="upper right", fontsize=12)
    sim_label(ax)
    save(fig, path, PAPER_FORMATS)


def paper_t0_phase(results, path):
    """Pooled pulls vs position of t0 inside the sample (nominal noise)."""
    names = ["F_prompt", "I_fast", "tau_slow", "I_slow"]
    t0s = sorted({r["truth"]["t0"] for r in results})
    phase = np.array([bin_phase_ns(t0) for t0 in t0s])
    sub = {t0: [r for r in results if r["truth"]["t0"] == t0 and noise_factor(r) == 1.0] for t0 in t0s}

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 2, 1.3], hspace=0.06))
    for k, name in enumerate(names):
        st = np.array([pull_stats(np.concatenate([pulls_of(r, name) for r in sub[t0]])) for t0 in t0s])
        kw = dict(marker=MARKERS[k], color=OI[k + 1], ms=8, capsize=4, lw=1.5, label=LABELS[name])
        axes[0].errorbar(phase + 0.15 * (k - 1.5), st[:, 0], st[:, 1], **kw)
        axes[1].errorbar(phase + 0.15 * (k - 1.5), st[:, 2], st[:, 3], **kw)
    fail = [100 * np.mean([1 - r["valid"].mean() for r in sub[t0]]) for t0 in t0s]
    axes[2].plot(phase, fail, "o-", color=OI[0], ms=8)

    axes[0].axhline(0, color="k", lw=1)
    axes[1].axhline(1, color="k", lw=1)
    axes[0].set_ylabel("Pull mean")
    axes[1].set_ylabel("Pull width")
    axes[2].set_ylabel("Failed [%]")
    axes[2].set_xlabel(r"$t_0$ position inside the sample [ns]")
    axes[2].set_xlim(-1, DT * 1e3)
    axes[2].set_ylim(bottom=0)
    axes[0].set_ylim(top=axes[0].get_ylim()[1] + 0.35 * np.ptp(axes[0].get_ylim()))
    axes[0].legend(loc="upper right", ncol=2, fontsize=13)
    sim_label(axes[0])
    save(fig, path, PAPER_FORMATS)


def paper_bias_over_resolution(results, summaries, path):
    """Bias in units of the statistical resolution (toy spread): the fit bias is
    negligible if |bias|/sigma_stat << 1."""
    names = ["tau_slow", "F_prompt", "I_fast", "I_slow"]
    if SCAN_MODE == "grid":
        nom = results[nominal_index(results)]
        sel = [i for i, r in enumerate(results)
               if noise_factor(r) == 1.0 and r["truth"]["t0"] == nom["truth"]["t0"]
               and r["truth"]["F_prompt_true"] == nom["truth"]["F_prompt_true"]]
        xval = lambda i: results[i]["truth"]["tau_slow"]
        xlabel = r"$\tau_{\rm slow}^{\rm true}$ [$\mu$s]"
    else:
        sel = list(range(len(results)))
        xval = lambda i: results[i]["truth"]["field_kV"]
        xlabel = "Drift HV [kV]"
    sel.sort(key=xval)
    x = np.array([xval(i) for i in sel])
    dx = 0.012 * np.ptp(x) if np.ptp(x) > 0 else 0.01

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhspan(-0.1, 0.1, color="0.9", zorder=0, label=r"$\pm$10% of $\sigma_{\rm stat}$")
    ax.axhline(0, color="k", lw=1)
    for k, name in enumerate(names):
        b = np.array([summaries[i][name]["bias"] / summaries[i][name]["spread"] for i in sel])
        e = np.array([summaries[i][name]["bias_err"] / summaries[i][name]["spread"] for i in sel])
        ax.errorbar(x + dx * (k - 1.5), b, e, marker=MARKERS[k], color=OI[k + 1], ms=8,
                    capsize=4, lw=1.5, label=LABELS[name])
    lim = max(0.5, 1.2 * np.nanmax(np.abs(ax.get_ylim())))
    ax.set_ylim(-lim, 1.6 * lim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Bias / $\sigma_{\rm stat}$")
    ax.legend(loc="upper right", ncol=2, fontsize=12)
    sim_label(ax)
    save(fig, path, PAPER_FORMATS)


def paper_resolution_vs_noise(results, summaries, path):
    """Relative statistical resolution vs noise level: toy spread (filled) and
    mean Minuit error (open) -- they must agree if the errors are calibrated."""
    names = ["tau_slow", "F_prompt", "I_slow"]
    nom = results[nominal_index(results)]
    sel = [i for i, r in enumerate(results)
           if r["truth"]["t0"] == nom["truth"]["t0"]
           and r["truth"]["F_prompt_true"] == nom["truth"]["F_prompt_true"]
           and r["truth"]["tau_slow"] == nom["truth"]["tau_slow"]]
    sel.sort(key=lambda i: noise_factor(results[i]))
    sig = np.array([NOISE_SIGMA * noise_factor(results[i]) for i in sel])

    fig, ax = plt.subplots(figsize=(8, 6))
    for k, name in enumerate(names):
        spread = np.array([summaries[i][name]["spread"] / abs(summaries[i][name]["true"]) for i in sel]) * 100
        merr = np.array([summaries[i][name]["mean_err"] / abs(summaries[i][name]["true"]) for i in sel]) * 100
        ax.plot(sig, spread, marker=MARKERS[k], color=OI[k + 1], ms=9, lw=0, label=LABELS[name])
        ax.plot(sig, merr, marker=MARKERS[k], color=OI[k + 1], ms=14, mfc="none", lw=1.2, ls="--")
    ax.plot([], [], "ko", ms=8, lw=0, label="Toy spread")
    ax.plot([], [], "ko--", ms=12, mfc="none", label="Mean fit error")
    ax.set_xlabel(r"Noise $\sigma$ [ADC]")
    ax.set_ylabel(r"Relative resolution [%]")
    ax.set_xlim(0, 1.1 * sig.max())
    ax.set_ylim(0, 1.5 * ax.get_ylim()[1])
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), ncol=2, fontsize=12)
    sim_label(ax)
    save(fig, path, PAPER_FORMATS)


def paper_pull_distributions(res, path):
    """Pull distributions of tau_slow and F_prompt at the nominal point."""
    names = ["tau_slow", "F_prompt"]
    x = np.linspace(-5, 5, 400)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, name in zip(axes, names):
        p = pulls_of(res, name)
        m, me, w, we = pull_stats(p)
        ax.hist(p, bins=40, range=(-5, 5), density=True, histtype="step", color=OI[0], lw=2,
                label=f"Toys ({p.size})")
        ax.plot(x, stats.norm.pdf(x), color=OI[2], ls="--", lw=2, label=r"$\mathcal{N}(0,1)$")
        ax.plot(x, stats.norm.pdf(x, m, w), color=OI[1], lw=2,
                label=rf"$\mu$ = {m:.2f} $\pm$ {me:.2f}" "\n" rf"$\sigma$ = {w:.2f} $\pm$ {we:.2f}")
        ax.set_xlabel(f"Pull {LABELS[name]}")
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 0.62)
        ax.legend(loc="upper right", fontsize=13)
        sim_label(ax)
    axes[0].set_ylabel("Probability density")
    fig.subplots_adjust(wspace=0.05)
    save(fig, path, PAPER_FORMATS)


def make_paper_figures(results, summaries, t, paper_dir):
    paper_dir.mkdir(parents=True, exist_ok=True)
    nom = results[nominal_index(results)]
    paper_example(nom, t, paper_dir / "fig1_example_toy")
    paper_pull_distributions(nom, paper_dir / "fig2_pull_distributions")
    paper_pull_summary(results, paper_dir / "fig3_pull_summary")
    paper_bias_over_resolution(results, summaries, paper_dir / "fig4_bias_over_resolution")
    if SCAN_MODE == "grid":
        if len({r["truth"]["t0"] for r in results}) > 1:
            paper_t0_phase(results, paper_dir / "fig5_t0_phase")
        if len({noise_factor(r) for r in results}) > 1:
            paper_resolution_vs_noise(results, summaries, paper_dir / "fig6_resolution_vs_noise")


# =============================================================================
# MAIN
# =============================================================================
def main():
    global N_TOYS
    parser = argparse.ArgumentParser(description="Toy MC for the forward-fold fit.")
    parser.add_argument("--n-toys", type=int, default=N_TOYS)
    parser.add_argument("--template", type=Path, default=TEMPLATE_PATH)
    args = parser.parse_args()
    N_TOYS = args.n_toys

    plots_dir = OUTPUT_DIR / "plots"
    data_dir = OUTPUT_DIR / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    template = load_template(args.template)
    model = make_model(template)
    t = np.arange(0.0, T_MAX, DT)
    rng = np.random.default_rng(RNG_SEED)

    points = build_grid() if SCAN_MODE == "grid" else TRUTH_POINTS
    csv_path = data_dir / "toy_mc_scan.csv"
    if csv_path.exists():
        csv_path.unlink()

    t_start = time.time()
    results, summaries = [], []
    for ip, truth in enumerate(points):
        elapsed = time.time() - t_start
        eta = elapsed / ip * (len(points) - ip) if ip else float("nan")
        print(f"[toy MC] point {ip + 1}/{len(points)}  {truth['label']}  "
              f"({N_TOYS} toys)  elapsed {elapsed / 60:.1f} min  ETA {eta / 60:.1f} min",
              flush=True)
        res = run_toys(truth, t, model, N_TOYS, rng)
        rows = summarize(res)
        results.append(res)
        summaries.append(rows)
        print(f"  valid: {res['valid'].mean():.3f}   at limit: {res['limit'].mean():.3f}")
        for name in (FREE_PARAMS + DERIVED if SCAN_MODE == "fields" else ["tau_slow", "F_prompt"]):
            r = rows[name]
            print(f"  {name:<9} rel.bias={r['rel_bias']:+.2e}  "
                  f"pull mu={r['pull_mean']:+.3f}  sigma={r['pull_width']:.3f}")

        lab = truth["label"]
        append_csv(res, rows, csv_path)
        if MAKE_POINT_PLOTS or SCAN_MODE == "fields":
            plot_example(res, t, plots_dir / f"example_toy_{lab}")
            plot_pulls(res, rows, plots_dir / f"pulls_{lab}")
            plot_rel_bias_hist(res, rows, plots_dir / f"deviation_{lab}")
            plot_chi2(res, plots_dir / f"chi2_{lab}")
            plot_correlation(res, plots_dir / f"correlation_{lab}")
        np.savez(data_dir / f"toys_{lab}.npz",
                 free_params=FREE_PARAMS, derived=DERIVED,
                 true_p=res["true_p"], true_d=res["true_d"],
                 est=res["est"], err=res["err"], d_est=res["d_est"], d_err=res["d_err"],
                 chi2=res["chi2"], ndof=res["ndof"], valid=res["valid"], limit=res["limit"])

    if SCAN_MODE == "grid":
        plot_grid(results, summaries, plots_dir)
    elif len(results) > 1:
        plot_summary_vs_field(results, summaries, plots_dir / "summary_vs_field")
    make_paper_figures(results, summaries, t, plots_dir / "paper")
    write_summary(results, summaries, data_dir / "toy_mc_summary.txt")
    print(f"[toy MC] total time {(time.time() - t_start) / 60:.1f} min")
    print(f"[toy MC] outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
