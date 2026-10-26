"""Fit normalized light yield using joint MC draws and correlated chi-square.

Run: python analysis/LY_vs_EF/fit_LY_new_cost.py
The zero-field reference is shared across fields within each MC trial.
Channel/sample combinations use independent reproducible random streams.
The covariance describes propagated charge uncertainty, not MC numerical error.
The fit uses a multivariate Gaussian approximation to the joint LY draws.
"""
from collections import defaultdict
from pathlib import Path
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dunestyle.matplotlib as dunestyle
from iminuit import Minuit

repo_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_dir))
from src.src import calc_integral_and_error, read_txt_error_file

TIMESTAMP = time.strftime("%Y%m%d_%H%M%S", time.localtime())
N_SAMPLES = 500_000
MC_SEED = 42
MAKE_DRAW_PLOTS = True

SAMPLE_LABELS = ['20260909_213507','20260909_125014']

CHANNELS = [2050,2051]

run_to_efield = {
    # Zero-field reference
    39510: 0.0,

    # Increasing HV scan
    39511: 0.028571,
    39512: 0.057143,
    39514: 0.085714,
    39515: 0.114286,
    39516: 0.142857,
    39517: 0.171429,
    39518: 0.200000,
    39519: 0.228571,
    39521: 0.257143,
    39522: 0.285714,
    39523: 0.314286,
    39525: 0.342857,
    39526: 0.371429,
    39527: 0.400000,
    39528: 0.428571,
    39529: 0.444857,

    # Decreasing HV scan
    39500: 0.444857,
    39501: 0.400000,
    39502: 0.342857,    
    39503: 0.285714,    
    39504: 0.228571,    
    39506: 0.171429,   
    39507: 0.114286,    
    39508: 0.057143,

    # Higher HV scan
    43380: 0.685,
    43381: 0.685,
    43383: 0.586,
    43384: 0.586,
    43386: 0.771,
    43387: 0.771,
    43389: 0.44,
    43390: 0.44,
    41523: 0.834,
    
}

E = [
    0.000,
    0.055,
    0.110,
    0.166,
    0.222,
    0.278,
    0.333,
    0.388,
    0.444,
    0.501,
]

PD_HD_result = [1.000,0.890,0.792,0.770,0.748,0.698,0.665,0.658,0.630,0.638]

PD_HD_err_low = [ 0.000, 0.022, 0.035, 0.025, 0.025, 0.018, 0.025, 0.030, 0.024, 0.029]

PD_HD_err_high = [ 0.000, 0.026, 0.038, 0.020, 0.025, 0.018, 0.023, 0.027, 0.022, 0.028]

relative_s1_previous = np.array([
    1.000000000000000,
    0.938679245283019,
    0.880261248185777,
    0.857764876632801,
    0.831640058055152,
    0.784470246734398,
    0.761248185776488,
    0.745283018867925,
    0.718432510885341,
    0.688679245283019,
    0.682148040638607,
    0.665457184325109,
    0.664731494920174,
    0.638606676342525,
    0.627721335268505,
    0.623367198838897,
    0.625544267053701,
    0.579100145137881,
])

efield_previous = np.array([
    0.000000,
    0.026250,
    0.054375,
    0.082500,
    0.110625,
    0.138125,
    0.166875,
    0.195625,
    0.221875,
    0.249375,
    0.276875,
    0.305625,
    0.333125,
    0.360625,
    0.388750,
    0.415625,
    0.431875,
    0.485625,
])

def Birks(E, B_1, k):
    E = np.asarray(E, dtype=float)
    return 1.0 - B_1 * E / (E + k)

def mc_uncertainty_norm(s_central, s_up, s_low, s0_central, s0_up, s0_low,
                        n_samples=500_000, seed=42, return_draws=False):
    """Normalize scalar/vector S by one shared S0; up/low inputs are bounds."""

    s_central = np.asarray(s_central, dtype=float)
    s_up = np.asarray(s_up, dtype=float)
    s_low = np.asarray(s_low, dtype=float)

    s0_central = np.asarray(s0_central, dtype=float).item()
    s0_up = np.asarray(s0_up, dtype=float).item()
    s0_low = np.asarray(s0_low, dtype=float).item()

    if s_central.ndim > 1 or s_central.size == 0:
        raise ValueError("S must be a scalar or nonempty 1-D array.")
    if any(not np.all(np.isfinite(value)) for value in
           (s_central, s_up, s_low, s0_central, s0_up, s0_low)):
        raise ValueError("Charges and bounds must be finite.")

    # Convert upper/lower bounds into error magnitudes.
    up = s_up - s_central
    down = s_central - s_low
    up0 = s0_up - s0_central
    down0 = s0_central - s0_low
    if np.any(up < 0) or np.any(down < 0) or up0 < 0 or down0 < 0:
        raise ValueError("Require lower <= central <= upper.")
    if s0_central <= 0:
        raise ValueError("The reference charge must be positive.")

    rng = np.random.default_rng(seed)
    # Draw each numerator independently.
    z = rng.standard_normal((n_samples, s_central.size))
    s_draws = s_central + np.where(z >= 0, up * z, down * z)

    # Draw S0 ONCE per trial and reuse it for every numerator.
    z0 = rng.standard_normal(n_samples)
    s0_draws = s0_central + np.where(z0 >= 0, up0 * z0, down0 * z0)
    if np.any(s0_draws <= 0):
        raise ValueError("Reference draws reached zero or below; reconsider its uncertainty model.")

    # Each column contains the trials for one relative-LY point.
    combined = s_draws / s0_draws[:, None]

    if return_draws:
        return combined

    low, central, high = np.quantile(
        combined, [0.15865525393145707, 0.5, 0.8413447460685429], axis=0
    )

    if s_central.ndim == 0:
        return central.item(), (central - low).item(), (high - central).item()
    return central, central - low, high - central


def calculate_study_XA(csv_suffix, channel, y_err_runs, path_waveforms, seed=42):
    """Return sorted fields, joint normalized draws and number of input runs."""
    records = []
    for csv_file in sorted(path_waveforms.glob(
        f"channel_{channel}_coincidence_scan_run_*_{csv_suffix}.csv"
    )):
        match = re.search(r"channel_(\d+)_coincidence_scan_run_(\d+)", csv_file.name)
        if not match:
            continue
        run = int(match.group(2))
        if run not in run_to_efield:
            print(f"Run {run:06d} has no electric-field mapping. Skipping.")
            continue
        waveform = pd.read_csv(csv_file)["mean"].to_numpy()
        lower, central, upper = calc_integral_and_error(
            waveform, y_err_runs[run][channel]
        )
        records.append((run, lower, central, upper))

    if not records:
        return np.array([], dtype=float), None, 0

    runs = np.array([record[0] for record in records])
    if np.unique(runs).size != runs.size:
        raise ValueError(f"Duplicate run measurements for channel {channel}")
    reference = np.flatnonzero(runs == 39510)
    if reference.size == 0:
        raise RuntimeError(f"Missing run 039510 reference for channel {channel}")
    ref = reference[0]
    lower, central, upper = np.array([record[1:] for record in records]).T

    draws = mc_uncertainty_norm(
        central, upper, lower, central[ref], upper[ref], lower[ref],
        n_samples=N_SAMPLES, seed=seed, return_draws=True,
    )
    draws[:, ref] = 1.0

    points_by_efield = defaultdict(list)
    for index, run in enumerate(runs):
        points_by_efield[run_to_efield[run]].append(index)
    efields = np.asarray(sorted(points_by_efield), dtype=float)
    # Average repeated runs within each trial to retain the shared reference.
    field_draws = np.column_stack([
        np.mean(draws[:, points_by_efield[efield]], axis=1)
        for efield in efields
    ])
    return efields, field_draws, len(records)


def make_birks_cost(efields, mean_rel_ly, cov_ly, fit_mask):
    """Build r.T @ inv(C) @ r using a Cholesky factor instead of an inverse."""
    x = efields[fit_mask]
    y = mean_rel_ly[fit_mask]
    cov = cov_ly[np.ix_(fit_mask, fit_mask)]
    # Zero field must be excluded because its exact normalization has no variance.
    factor = np.linalg.cholesky(cov)

    def cost_birks(B_1, k):
        residual = y - Birks(x, B_1, k)
        whitened = np.linalg.solve(factor, residual)
        return float(whitened @ whitened)

    cost_birks.errordef = Minuit.LEAST_SQUARES
    cost_birks.ndata = len(x)
    return cost_birks


def main():
    output_dir = Path(__file__).resolve().parent
    stem = f"fit_LY_new_cost_chs_{'_'.join(map(str, CHANNELS))}_{TIMESTAMP}"
    seeds = iter(np.random.SeedSequence(MC_SEED).spawn(
        len(SAMPLE_LABELS) * len(CHANNELS)
    ))
    efields = None
    mean_draws = None
    used_channels = []
    used_samples = []

    for sample_label in SAMPLE_LABELS:
        path_waveforms = repo_dir / "coincidence/selected_waveforms" / sample_label
        for channel in CHANNELS:
            current_fields, draws, count = calculate_study_XA(
                sample_label, channel, read_txt_error_file(sample_label, channel),
                path_waveforms, seed=next(seeds),
            )
            if count == 0:
                print(f"No data found for channel {channel} in sample {sample_label}. Skipping.")
                continue
            if efields is None:
                efields = current_fields.copy()
                mean_draws = draws
            else:
                if current_fields.shape != efields.shape or not np.allclose(current_fields, efields):
                    raise ValueError(
                        f"Electric-field values differ for channel {channel}, sample {sample_label}"
                    )
                mean_draws += draws
            used_channels.append(channel)
            used_samples.append(sample_label)

    if mean_draws is None:
        raise RuntimeError("No valid data found.")

    # Equal-weight mean of independent channel/sample combinations in each trial.
    mean_draws /= len(used_channels)
    cov_ly = np.cov(mean_draws, rowvar=False, ddof=1)
    # Do not divide by the MC trial count: these draws propagate measurement error.
    lower, mean_rel_ly, upper = np.quantile(
        mean_draws, [0.15865525393145707, 0.5, 0.8413447460685429], axis=0
    )
    err_low = mean_rel_ly - lower
    err_up = upper - mean_rel_ly
    fit_mask = (
        (efields > 0.0)
        & np.isfinite(mean_rel_ly)
        & np.isfinite(np.diag(cov_ly))
        & (np.diag(cov_ly) > 0.0)
    )
    if np.count_nonzero(fit_mask) <= 2:
        raise RuntimeError("Birks fit needs at least three points with nonzero variance.")

    cost_birks = make_birks_cost(efields, mean_rel_ly, cov_ly, fit_mask)
    fit = Minuit(cost_birks, B_1=1.0, k=0.1)
    fit.migrad()
    fit.hesse()
    if not fit.valid or not fit.accurate:
        raise RuntimeError(f"Birks fit or parameter covariance is invalid:\n{fit}")
    ndof = np.count_nonzero(fit_mask) - fit.nfit
    print(fit)

    # Save the covariance and its field ordering for inspection and reuse.
    np.savez(
        output_dir / f"{stem}.npz",
        efields=efields, mean_rel_ly=mean_rel_ly,
        err_low=err_low, err_up=err_up, cov_ly=cov_ly, fit_mask=fit_mask,
        fit_parameter_names=np.array(list(fit.parameters)),
        fit_values=np.array(list(fit.values)), fit_errors=np.array(list(fit.errors)),
        fit_covariance=np.asarray(fit.covariance), chi2=fit.fval, ndof=ndof,
        channels=np.array(used_channels), samples=np.array(used_samples),
        n_samples=N_SAMPLES, seed=MC_SEED,
    )

    if MAKE_DRAW_PLOTS:
        for index in np.flatnonzero(efields > 0.0):
            fig, ax = plt.subplots(dpi=150)
            ax.hist(mean_draws[:, index], bins=60, alpha=0.7)
            for value, color, label in (
                (lower[index], "red", "Lower"),
                (mean_rel_ly[index], "black", "Central"),
                (upper[index], "green", "Upper"),
            ):
                ax.axvline(value, color=color, ls="--", label=f"{label}: {value:.4f}")
            ax.set_xlabel(r"$S/S_0$ Relative Light Yield")
            ax.set_ylabel("MC Draw (Counts/bin)")
            ax.set_title(f"Electric field: {efields[index]:g} kV/cm")
            ax.legend()
            field_label = f"{efields[index]:g}".replace(".", "p")
            fig.savefig(output_dir / f"{stem}_mc_{field_label}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    del mean_draws
    x_fit = np.linspace(0, 0.9, 200)
    fig, ax = plt.subplots(dpi=150)
    ax.grid()
    ax.plot(x_fit, Birks(x_fit, *fit.values), color="red", ls="--", lw=2, label="Birks fit")
    ax.scatter(efield_previous, relative_s1_previous, marker="x", color="gray", label="M8 previous study")
    ax.errorbar(E, PD_HD_result, yerr=[PD_HD_err_low, PD_HD_err_high],
                color="gray", fmt=".", capsize=1, label="PD-HD Data")
    ax.errorbar(efields, mean_rel_ly, yerr=[err_low, err_up], fmt=".",
                label=f"Mean Channels {list(dict.fromkeys(used_channels))}")
    fit_text = (
        "Birks fit with covariance\n"
        rf"$B_1 = {fit.values['B_1']:.3f} \pm {fit.errors['B_1']:.3f}$" "\n"
        rf"$k_\epsilon = {fit.values['k']:.3f} \pm {fit.errors['k']:.3f}$" "\n"
        rf"$\chi^2/\mathrm{{ndof}} = {fit.fval:.1f}/{ndof} = {fit.fval / ndof:.2f}$"
    )
    ax.text(0.64, 0.95, fit_text, transform=ax.transAxes, va="top", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85})
    ax.set_xlabel("Electric field (kV/cm)")
    ax.set_ylabel(r"$S/S_0$ (LY Normalized)")
    ax.set_ylim(0.3, 1.2)
    dunestyle.Preliminary(x=0.05, y=0.9, ax=ax)
    dunestyle.WIP(x=0.05, y=0.83, ax=ax)
    ax.legend(loc="lower left", frameon=True, fontsize=8)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
