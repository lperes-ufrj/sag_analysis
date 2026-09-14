"""Fit relative LY with MC propagation of the supplied charge uncertainties.

Usage: python analysis/LY_vs_EF/fit_LY_mc_unc.py
Runs are sampled independently before sharing a reference in normalization.
The fits use marginal errors, without covariance between electric-field points.
The charge-uncertainty model is provided by src.calc_integral_and_error.
"""
from pathlib import Path
from collections import defaultdict
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import re
import time
import dunestyle.matplotlib as dunestyle

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from iminuit import Minuit
from iminuit.cost import LeastSquares



repo_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_dir))
from src.src import calc_integral_and_error,read_txt_error_file


SAMPLE_LABELS = ['20260909_213507','20260909_125014']
CHANNELS = [2050,2051,2060,2061,2080,2081]
script_dir = Path(__file__).resolve().parent.parent
#print(f"Script directory: {script_dir}")
repo_dir = script_dir.parent
path_templates = "../" / repo_dir / "filter/templates_large_pulses"

templates_ch_1010_charge = np.trapezoid(np.loadtxt(path_templates / "template_42228_C1_1.txt"))
templates_ch_1011_charge = np.trapezoid(np.loadtxt(path_templates / "template_42228_C1_2.txt"))
templates_ch_1020_charge = np.trapezoid(np.loadtxt(path_templates / "template_41519_C2_1.txt"))
templates_ch_1021_charge = np.trapezoid(np.loadtxt(path_templates / "template_41519_C2_2.txt"))
templates_ch_1030_charge = np.trapezoid(np.loadtxt(path_templates / "template_41536_C3_1.txt"))
templates_ch_1031_charge = np.trapezoid(np.loadtxt(path_templates / "template_41536_C3_2.txt"))
templates_ch_1040_charge = np.trapezoid(np.loadtxt(path_templates / "template_42067_C4_1.txt"))
templates_ch_1041_charge = np.trapezoid(np.loadtxt(path_templates / "template_42067_C4_2.txt"))
templates_ch_1050_charge = np.trapezoid(np.loadtxt(path_templates / "template_42228_C5_1.txt"))
templates_ch_1051_charge = np.trapezoid(np.loadtxt(path_templates / "template_42228_C5_2.txt"))
templates_ch_1060_charge = np.trapezoid(np.loadtxt(path_templates / "template_40807_C6_1.txt"))
templates_ch_1061_charge = np.trapezoid(np.loadtxt(path_templates / "template_40807_C6_2.txt"))
templates_ch_1070_charge = np.trapezoid(np.loadtxt(path_templates / "template_40808_C7_1.txt"))
templates_ch_1071_charge = np.trapezoid(np.loadtxt(path_templates / "template_40808_C7_2.txt"))
templates_ch_1080_charge = np.trapezoid(np.loadtxt(path_templates / "template_40808_C8_1.txt"))
templates_ch_1081_charge = np.trapezoid(np.loadtxt(path_templates / "template_40808_C8_2.txt"))

templates_ch_2010_charge = np.trapezoid(np.loadtxt(path_templates / "template_42379_M1_1.txt"))
templates_ch_2011_charge = np.trapezoid(np.loadtxt(path_templates / "template_42379_M1_2.txt"))
# Não existe template M2_1 para o canal 2020.
templates_ch_2021_charge = np.trapezoid(np.loadtxt(path_templates / "template_42379_M2_2.txt"))
templates_ch_2030_charge = np.trapezoid(np.loadtxt(path_templates / "template_40801_M3_1.txt"))
templates_ch_2031_charge = np.trapezoid(np.loadtxt(path_templates / "template_40801_M3_2.txt"))
templates_ch_2040_charge = np.trapezoid(np.loadtxt(path_templates / "template_40989_M4_1.txt"))
templates_ch_2041_charge = np.trapezoid(np.loadtxt(path_templates / "template_40989_M4_2.txt"))
templates_ch_2050_charge = np.trapezoid(np.loadtxt(path_templates / "template_42320_M5_1.txt"))
templates_ch_2051_charge = np.trapezoid(np.loadtxt(path_templates / "template_42320_M5_2.txt"))
templates_ch_2060_charge = np.trapezoid(np.loadtxt(path_templates / "template_40808_M6_1.txt"))
templates_ch_2061_charge = np.trapezoid(np.loadtxt(path_templates / "template_40808_M6_2.txt"))
templates_ch_2070_charge = np.trapezoid(np.loadtxt(path_templates / "template_43229_M7_1.txt"))
templates_ch_2071_charge = np.trapezoid(np.loadtxt(path_templates / "template_43229_M7_2.txt"))
templates_ch_2080_charge = np.trapezoid(np.loadtxt(path_templates / "template_42321_M8_1.txt"))
templates_ch_2081_charge = np.trapezoid(np.loadtxt(path_templates / "template_42321_M8_2.txt"))


list_templates_charge = {
    1010: templates_ch_1010_charge,
    1011: templates_ch_1011_charge,
    1020: templates_ch_1020_charge,
    1021: templates_ch_1021_charge,
    1030: templates_ch_1030_charge,
    1031: templates_ch_1031_charge,
    1040: templates_ch_1040_charge,
    1041: templates_ch_1041_charge,
    1050: templates_ch_1050_charge,
    1051: templates_ch_1051_charge,
    1060: templates_ch_1060_charge,
    1061: templates_ch_1061_charge,
    1070: templates_ch_1070_charge,
    1071: templates_ch_1071_charge,
    1080: templates_ch_1080_charge,
    1081: templates_ch_1081_charge,

    2010: templates_ch_2010_charge,
    2011: templates_ch_2011_charge,
    2021: templates_ch_2021_charge,
    2030: templates_ch_2030_charge,
    2031: templates_ch_2031_charge,
    2040: templates_ch_2040_charge,
    2041: templates_ch_2041_charge,
    2050: templates_ch_2050_charge,
    2051: templates_ch_2051_charge,
    2060: templates_ch_2060_charge,
    2061: templates_ch_2061_charge,
    2070: templates_ch_2070_charge,
    2071: templates_ch_2071_charge,
    2080: templates_ch_2080_charge,
    2081: templates_ch_2081_charge,
}



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

# Approximate ProtoDUNE-VD M8 points digitized from the figure
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

def formatar_nome_arquivo(texto):
    # Substitui os espaços por '_' e os pontos por 'p'
    texto_limpo = texto.replace(' ', '_').replace('.', 'p').replace('/','_div_')
    return texto_limpo


def Calc_Rel_LY(charge, template_charge):
    return charge / template_charge


def LArQL(E_D, B_1, k_e, B_2, E_0):
    E_D = np.abs(np.asarray(E_D, dtype=float))
    return 1.0 - B_1 * E_D / (E_D + k_e) + B_2 * (-np.expm1(-E_D / E_0))

def Birks(E, B_1, k):
    E = np.asarray(E, dtype=float)
    return 1.0 - B_1 * E / (E + k)

def mc_central_uncertainty(x, err_minus, err_plus,
                        weights=None, n_samples=500_000, seed=42, make_draw_plots= False, label=None):
    x = np.asarray(x, dtype=float)
    down = np.asarray(err_minus, dtype=float)
    up = np.asarray(err_plus, dtype=float)

    # Equal weights by default.
    a = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    a = a / a.sum()

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_samples, len(x)))

    shifts = np.where(z >= 0, up * z, down * z)
    combined = (x + shifts) @ a

    low, central, high = np.quantile(
        combined, [0.15865525393145707, 0.5, 0.8413447460685429]
    )

    if make_draw_plots == True:
        fig, ax = plt.subplots(dpi=150)
        ax.hist(combined, bins=60, alpha=0.7, label=label)

        ax.axvline(central, color="black", linewidth=2,
                   label=f"Central: {central:.4f}")
        ax.axvline(low, color="red", linestyle="--",
                   label=f"Lower: {low:.4f}")
        ax.axvline(high, color="green", linestyle="--",
                   label=f"Upper: {high:.4f}")

        ax.legend()
        ax.set_ylabel("MC Draw (Counts/bin)")
        ax.set_xlabel(rf"$S/S_0$ Relative Light Yield")
        fig.savefig(
            repo_dir / "analysis" / "LY_vs_EF"
            / f"fit_LY_mc_fit_{formatar_nome_arquivo(label)}_{'_'.join(map(str, CHANNELS))}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    return central, central - low, high - central

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

def calculate_study_XA(csv_suffix, channel, y_err_runs,path_waveforms):
    csv_files = sorted(
        path_waveforms.glob(f"channel_*run*{csv_suffix}.csv")
    )

    
    reference_charge_lower = {}
    reference_charge_central = {}
    reference_charge_upper = {}

    charge_records = []
    ch_found = 0
    for csv_file in csv_files:
        match = re.search( r"channel_(\d+)_coincidence_scan_run_(\d+)", csv_file.name)
        
        if not match:
            continue
        #print(f"Processing file: {csv_file.name}")
        ch = int(match.group(1))
        if ch != channel:
            continue

        run = int(match.group(2))

        if run not in run_to_efield:
            print(f"Run {run:06d} has no electric-field mapping. Skipping.")
            continue
        if ch not in list_templates_charge:
            raise KeyError(f"Channel {ch} has no charge template")

        mean_waveform = pd.read_csv(csv_file)["mean"].to_numpy()
        charge_lower, charge_central, charge_upper = calc_integral_and_error(mean_waveform,y_err_runs[run][ch])
        ly_central = Calc_Rel_LY(charge_central, list_templates_charge[ch])
        ly_err_up = Calc_Rel_LY(charge_upper, list_templates_charge[ch]) 
        ly_err_low = Calc_Rel_LY(charge_lower, list_templates_charge[ch])
        
        charge_records.append((run, ch, ly_err_low, ly_central, ly_err_up))
        ch_found += 1

        if run == 39510:
            reference_charge_central[ch] = ly_central
            reference_charge_lower[ch] = ly_err_low
            reference_charge_upper[ch] = ly_err_up

    if not charge_records:
        empty = np.array([], dtype=float)
        return empty, empty.copy(), empty.copy(), empty.copy(), 0

    if channel not in reference_charge_central:
        raise RuntimeError(f"Missing run 039510 reference for channel {channel}")

    runs = np.array([record[0] for record in charge_records])
    if np.unique(runs).size != runs.size:
        raise ValueError(f"Duplicate run measurements for channel {channel}")
    lower = np.array([record[2] for record in charge_records])
    central = np.array([record[3] for record in charge_records])
    upper = np.array([record[4] for record in charge_records])

    draws = mc_uncertainty_norm(
        central, upper, lower,
        reference_charge_central[channel],
        reference_charge_upper[channel],
        reference_charge_lower[channel],
        return_draws=True,
    )

    # Reference divided by itself is exactly one.
    draws[:, runs == 39510] = 1.0

    points_by_efield = defaultdict(list)

    for i, run in enumerate(runs):
        points_by_efield[run_to_efield[run]].append(draws[:, i])

    efields = np.asarray(sorted(points_by_efield), dtype=float)
    results = []

    for efield in efields:
        # Average repeated runs within each MC trial.
        combined = np.mean(points_by_efield[efield], axis=0)

        low, central, high = np.quantile(
            combined,
            [0.15865525393145707, 0.5, 0.8413447460685429],
        )

        results.append((central, low, high))

    central, lower, upper = np.asarray(results, dtype=float).T

    return efields, central, upper, lower, ch_found



def main() -> None:

    mean_rel_ly = []
    mean_rel_ly_err_up = []
    mean_rel_ly_err_low = []
    efields = None
        
    for sample_label in SAMPLE_LABELS:
        
        path_waveforms = (
        repo_dir / "coincidence/selected_waveforms" / sample_label
        )
        #print(f"Path to waveforms: {path_waveforms}")

        for ch in CHANNELS:
            y_err = read_txt_error_file(sample_label, ch)
            #print(f"y_err: {y_err}")
            current_efields, rel_ly, rel_ly_err_up, rel_ly_err_low, ch_found  = calculate_study_XA(sample_label, ch, y_err,path_waveforms)
            if ch_found == 0:
                print(f"No data found for channel {ch} in sample {sample_label}. Skipping.")
                continue

            if rel_ly.size == 0:
                print(f"No data found for channel {ch} in sample {sample_label}. Skipping.")
                continue

            if efields is None:
                efields = current_efields.copy()
            elif (
                current_efields.shape != efields.shape
                or not np.allclose(current_efields, efields)
            ):
                raise ValueError(
                    f"Electric-field values differ for channel {ch}, sample {sample_label}"
                )

            mean_rel_ly.append(rel_ly)
            mean_rel_ly_err_up.append(rel_ly_err_up)
            mean_rel_ly_err_low.append(rel_ly_err_low)

            #print(rel_ly, rel_ly_err_up, rel_ly_err_low)
    if efields is None or not mean_rel_ly:
        raise RuntimeError("No valid data found.")

    # Shape: (number of channel/sample combinations, number of fields).
    central_by_channel = np.stack(mean_rel_ly)
    lower_by_channel = np.stack(mean_rel_ly_err_low)
    upper_by_channel = np.stack(mean_rel_ly_err_up)

    # calculate_study_XA returns bounds; MC requires error magnitudes.
    down_by_channel = central_by_channel - lower_by_channel
    up_by_channel = upper_by_channel - central_by_channel

    if (
        not np.all(np.isfinite(central_by_channel))
        or not np.all(np.isfinite(down_by_channel))
        or not np.all(np.isfinite(up_by_channel))
        or np.any(down_by_channel < 0)
        or np.any(up_by_channel < 0)
    ):
        raise ValueError("Invalid channel values or uncertainty bounds.")

    results = []

    for i, efield in enumerate(efields):
        if efield == 0.0:
            # Reference divided by itself: exactly one, with zero uncertainty.
            results.append((1.0, 0.0, 0.0))
            continue

        results.append(
            mc_central_uncertainty(
                central_by_channel[:, i],
                down_by_channel[:, i],
                up_by_channel[:, i],
                seed=42 + i,
                make_draw_plots= True,
                label=rf"{efield} kV/cm"
            )
        )

    mean_rel_ly, err_low, err_up = np.asarray(results, dtype=float).T

    # Preserve the bounds convention used by your existing plotting code.
    mean_rel_ly_err_low = mean_rel_ly - err_low
    mean_rel_ly_err_up = mean_rel_ly + err_up

    # Symmetric approximation for the existing LeastSquares fits.
    err = 0.5 * (err_low + err_up)

    fit_mask = (
        (efields > 0.0)
        & np.isfinite(efields)
        & np.isfinite(mean_rel_ly)
        & np.isfinite(err)
        & (err > 0.0)
    )

    if np.count_nonzero(fit_mask) <= 4:
        raise ValueError("Need at least five valid points for the LArQL fit.")
    cost = LeastSquares(
    efields[fit_mask],
    mean_rel_ly[fit_mask],
    err[fit_mask],
    LArQL,
    )

    m = Minuit(cost, 
           B_1 = 0.92,
           B_2 = 0.41,
           E_0 = 0.05,
           k_e = 0.07)

    m.interactive()
    m.migrad()
    m.hesse()

    ndof = np.count_nonzero(fit_mask) - m.nfit
    reduced_chi2 = m.fval / ndof
    x_fit = np.linspace(0, efields.max(), 500)

    y_fit = LArQL(
        x_fit,
        m.values["B_1"],
        m.values["k_e"],
        m.values["B_2"],
        m.values["E_0"],
    )

    cost_birks = LeastSquares(
    efields[fit_mask],
    mean_rel_ly[fit_mask],
    err[fit_mask],
    Birks,
    )


    m_birks = Minuit(cost_birks, 
           B_1 = 1.0,
           k = 0.1,
            )

    m_birks.interactive()
    m_birks.migrad()
    m_birks.hesse()

    ndof_birks = np.count_nonzero(fit_mask) - m_birks.nfit
    reduced_chi2_birks = m_birks.fval / ndof_birks

    y_fit_birks = Birks(
        x_fit,
        m_birks.values["B_1"],
        m_birks.values["k"],
    )

    

    fit_text_larql = (
        rf"$\mathbf{{Fit\ LArQL:}}$"
        "\n"
        rf"$B_1 = {m.values['B_1']:.3f} \pm {m.errors['B_1']:.3f}$"
        "\n"
        rf"$k_\epsilon = {m.values['k_e']:.3f} \pm {m.errors['k_e']:.3f}$"
        "\n"
        rf"$B_2 = {m.values['B_2']:.3f} \pm {m.errors['B_2']:.3f}$"
        "\n"
        rf"$E_0 = {m.values['E_0']:.3f} \pm {m.errors['E_0']:.3f}$"
        "\n"
        rf"$\chi^2/\mathrm{{ndof}} = {m.fval:.1f}/{ndof}"
        rf" = {reduced_chi2:.2f}$"
        )

    fit_text_birks = (
        rf"$\mathbf{{Fit\ Birks:}}$"
        "\n"
        rf"$B_1 = {m_birks.values['B_1']:.3f} \pm {m_birks.errors['B_1']:.3f}$"
        "\n"
        rf"$k_\epsilon = {m_birks.values['k']:.3f} \pm {m_birks.errors['k']:.3f}$"
        "\n"
        rf"$\chi^2/\mathrm{{ndof}} = {m_birks.fval:.1f}/{ndof_birks}"
        rf" = {reduced_chi2_birks:.2f}$"
        )

    plt.figure(dpi=150)
    plt.grid()
    plt.text(
        0.97,
        0.95,
        fit_text_larql,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    plt.text(
        0.97,
        0.63,
        fit_text_birks,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    plt.plot(x_fit, y_fit, color="blue", linewidth=2, label="LArQL fit")
    plt.plot(x_fit, y_fit_birks, color="red", ls='--',linewidth=2, label="Birks fit")
    plt.scatter(
        efield_previous,
        relative_s1_previous,
        marker="x",
        alpha=0.9,
        color='gray',
        label="M8 previous study",
    )
    # ProtoDUNE-HD reference
    plt.errorbar(
        E,
        PD_HD_result,
        yerr=[
            PD_HD_err_low,
            PD_HD_err_high,
        ],
        color="gray",
        fmt=".",
        capsize=1,
        label="PD-HD Data",
        zorder=3,
    )
    plt.errorbar(
        efields,
        mean_rel_ly,
        yerr=[mean_rel_ly - mean_rel_ly_err_low, mean_rel_ly_err_up - mean_rel_ly],
        fmt=".",
        label=f"Mean Channels {CHANNELS}",
    )
    plt.xlabel("Electric field (kV/cm)")
    plt.ylabel(rf"$S/S_0$ (LY Normalized)")

    plt.ylim([0.3,1.2])
    dunestyle.Preliminary(x=0.05,y=0.9)
    dunestyle.WIP(x=0.05,y=0.83)
    plt.legend(loc="lower left",ncols=1 ,frameon=True, fontsize=8)
    plt.savefig(
        repo_dir / "analysis" / "LY_vs_EF" / f"fit_LY_chs_{'_'.join(map(str,CHANNELS))}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

if __name__ == "__main__":
    main()
