"""Fit relative LY for one channel using errors on the mean waveform.

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


SAMPLE_LABELS = ['20261001_142744']
CHANNELS = [2080,2081]
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


run_to_Xeppm = {
    43440 : 0.01,
    43552 : 1.0,
    43717 : 2.0,
    43790 : 3.0,
    43903 : 5.0,
    44010 : 7.0,
    44108 : 10.0
}

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




def Calc_Rel_LY(charge, template_charge):
    return charge / template_charge


def LArQL(E_D, B_1, k_e, B_2, E_0):
    E_D = np.abs(np.asarray(E_D, dtype=float))
    return 1.0 - B_1 * E_D / (E_D + k_e) + B_2 * (-np.expm1(-E_D / E_0))

def Birks(E, B_1, k):
    E = np.asarray(E, dtype=float)
    return 1.0 - B_1 * E / (E + k)


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

        ch_found+=1
        run = int(match.group(2))

        if run not in run_to_Xeppm:
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

        if run == 39510:
            reference_charge_central[ch] = ly_central
            reference_charge_lower[ch] = ly_err_up
            reference_charge_upper[ch] = ly_err_low

    references = len(reference_charge_central)
    #print(references)
    if not references:
        print(
            f"Missing run 039510 reference for channels {sorted(reference_charge_central.keys())}"
        )

    points_by_efield_central = defaultdict(list)
    points_by_efield_err_up = defaultdict(list)
    points_by_efield_err_low = defaultdict(list)
    for run, channel, ly_err_low, ly_central, ly_err_up in charge_records:
        points_by_efield_central[run_to_efield[run]].append(
            ly_central / reference_charge_central[channel]
        )
        points_by_efield_err_up[run_to_efield[run]].append(
            ly_err_up / reference_charge_central[channel]
        )
        points_by_efield_err_low[run_to_efield[run]].append(
            ly_err_low / reference_charge_central[channel]
        )

    efields = np.asarray(sorted(points_by_efield_central), dtype=float)
   # print(f"Points HV: {points_by_efield_central}")

    means_rel_ly_central = np.asarray(
        [np.mean(points_by_efield_central[efield]) for efield in efields],
        dtype=float,
    )
    means_rel_ly_err_up = np.asarray(
        [np.mean(points_by_efield_err_up[efield]) for efield in efields],
        dtype=float,
    )
    means_rel_ly_err_low = np.asarray(
        [np.mean(points_by_efield_err_low[efield]) for efield in efields],
        dtype=float,
    )
    return efields, means_rel_ly_central, means_rel_ly_err_up, means_rel_ly_err_low, ch_found




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
           # print(f"Channel {ch}:")
            current_efields, rel_ly, rel_ly_err_up, rel_ly_err_low, ch_found  = calculate_study_XA(sample_label, ch, y_err,path_waveforms)
           # print(f"***********************************")
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
    mean_rel_ly = np.mean(np.array(mean_rel_ly), axis=0)
    mean_rel_ly_err_up = np.mean(np.array(mean_rel_ly_err_up), axis=0)
    mean_rel_ly_err_low = np.mean(np.array(mean_rel_ly_err_low), axis=0)

    #print(mean_rel_ly, mean_rel_ly_err_up, mean_rel_ly_err_low)
    fit_mask = (
        (efields > 0.0)
        & np.isfinite(efields)
        & np.isfinite(mean_rel_ly)
        & np.isfinite(mean_rel_ly_err_up)
        & (mean_rel_ly_err_up > 0.0)
    )


    err_low = mean_rel_ly - mean_rel_ly_err_low
    err_up = mean_rel_ly_err_up - mean_rel_ly

    err = (err_low + err_up) / 2  # Half-width
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

    ndof_birks = np.count_nonzero(fit_mask) - m.nfit
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
        rf"$B_1 = {m_birks.values['B_1']:.3f} \pm {m_birks._errors['B_1']:.3f}$"
        "\n"
        rf"$k_\epsilon = {m_birks.values['k']:.3f} \pm {m_birks.errors['k']:.3f}$"
        "\n"
        rf"$\chi^2/\mathrm{{ndof}} = {m_birks.fval:.1f}/{ndof_birks}"
        rf" = {reduced_chi2_birks:.2f}$"
        )

    plt.figure(dpi=150)
    plt.text(
        0.95,
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
        0.95,
        0.62,
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
        label=f"This study, channels: {CHANNELS}",
    )
    plt.xlabel("Electric field (kV/cm)")
    plt.ylabel(rf"$S/S_0$ (LY Normalized)")


    plt.ylim([0.3,1.2])
    dunestyle.Preliminary(x=0.05,y=0.9)
    dunestyle.WIP(x=0.05,y=0.83)
    plt.legend(loc="lower left",ncols=1 ,frameon=False, fontsize=8)
    plt.savefig(
        repo_dir / "analysis" / "LY_vs_EF" / f"fit_LY_one_ch_{'_'.join(map(str,CHANNELS))}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

if __name__ == "__main__":
    main()
