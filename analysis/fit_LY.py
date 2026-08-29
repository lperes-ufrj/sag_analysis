from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import re
import time
import dunestyle.matplotlib as dunestyle


sample_label = '20260828_134009'
script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parent
path_templates = repo_dir / "filter/templates_large_pulses"

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


path_waveforms = (
    repo_dir / "coincidence/selected_waveforms" / sample_label
)

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


def Calc_Charge(waveform, template_charge):
    return np.trapezoid(waveform[50:500]) / template_charge


def calculate_study_XA(csv_suffix, channels):
    csv_files = sorted(
        path_waveforms.glob(f"channel_*run*{csv_suffix}")
    )

    print(csv_files)
    reference_charge = {}
    charge_records = []

    for csv_file in csv_files:
        match = re.search(r"channel_(\d+)_run_(\d+)", csv_file.name)
        if not match:
            continue

        channel = int(match.group(1))
        run = int(match.group(2))

        if run not in run_to_efield:
            print(f"Run {run:06d} has no electric-field mapping. Skipping.")
            continue
        if channel not in list_templates_charge:
            raise KeyError(f"Channel {channel} has no charge template")

        mean_waveform = pd.read_csv(csv_file)["mean"].to_numpy()
        charge = Calc_Charge(mean_waveform, list_templates_charge[channel])
        charge_records.append((run, channel, charge))
        if run == 39510:
            reference_charge[channel] = charge

    missing_references = channels.difference(reference_charge)
    if missing_references:
        raise RuntimeError(
            f"Missing run 039510 reference for channels {sorted(missing_references)}"
        )

    points_by_efield = defaultdict(list)
    for run, channel, charge in charge_records:
        points_by_efield[run_to_efield[run]].append(
            charge / reference_charge[channel]
        )

    efields = np.asarray(sorted(points_by_efield), dtype=float)
    means = np.asarray(
        [np.mean(points_by_efield[efield]) for efield in efields],
        dtype=float,
    )
    return efields, means

def calculate_study_mean_all(datasets):
    reference_run = 39510
    reference_charge = {}
    charge_records = []
    requested_channels = set()

    for dataset in datasets:
        csv_suffix = dataset["csv_suffix"]
        channels = set(dataset["channels"])
        requested_channels.update(channels)

        csv_files = sorted(
            path_waveforms.glob(f"channel_*run*{csv_suffix}")
        )

        for csv_file in csv_files:
            match = re.search(
                r"channel_(\d+).*?run_?(\d+)",
                csv_file.name,
            )
            if not match:
                continue

            channel = int(match.group(1))
            run = int(match.group(2))

            if channel not in channels:
                continue

            if run not in run_to_efield:
                print(
                    f"Run {run:06d} has no electric-field mapping. "
                    "Skipping."
                )
                continue

            if channel not in list_templates_charge:
                raise KeyError(
                    f"Channel {channel} has no charge template"
                )

            mean_waveform = pd.read_csv(csv_file)["mean"].to_numpy()

            charge = Calc_Charge(
                mean_waveform,
                list_templates_charge[channel],
            )

            charge_records.append((run, channel, charge))

            if run == reference_run:
                reference_charge[channel] = charge

    missing_references = requested_channels.difference(reference_charge)

    if missing_references:
        raise RuntimeError(
            "Missing run 039510 reference for channels "
            f"{sorted(missing_references)}"
        )

    points_by_efield = defaultdict(list)

    for run, channel, charge in charge_records:
        normalized_charge = charge / reference_charge[channel]
        efield = run_to_efield[run]
        points_by_efield[efield].append(normalized_charge)

    efields = np.asarray(sorted(points_by_efield), dtype=float)

    means = np.asarray([
        np.mean(points_by_efield[efield])
        for efield in efields
    ], dtype=float)

    sems = np.asarray([
        (
            np.std(points_by_efield[efield], ddof=1)
            / np.sqrt(len(points_by_efield[efield]))
            if len(points_by_efield[efield]) > 1
            else 0.0
        )
        for efield in efields
    ], dtype=float)

    return efields, means, sems

study_configs_datasets = [
    {
        "label": "This study — mean all channels for M5, M6 and M8 ",
        "datasets": [
            {
                "csv_suffix": f"{sample_label}.csv",
                "channels": {2060,2061,2050,2051},
            },
            #{
            #    "csv_suffix": (
            #        "coinc_2030-2031-2040-2041_vs_2050-2051-2060-2061_"
            #        "save_2080-2081_"
            #        "window_10_ticks_min_amplitude_0_adc.csv"
            #    ),
            #    "channels": {2080, 2081},
            #},
        ],
    },
]




import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from iminuit import Minuit
from iminuit.cost import LeastSquares

def LArQL(E_D, B_1, k_e, B_2, E_0):
    E_D = np.abs(np.asarray(E_D, dtype=float))

    return (
        1.0
        - B_1 * E_D / (E_D + k_e)
        + B_2 * (1.0 - np.exp(-E_D / E_0))
    )

def Birks(E_D, B_1, k_e):
    E_D = np.abs(np.asarray(E_D, dtype=float))

    return (
        1.0
        - B_1 * E_D / (E_D + k_e)
    )



study_results = []

for config in study_configs_datasets:
    efields, means, sems = calculate_study_mean_all(
        config["datasets"]
    )
    study_results.append({
        **config,
        "efields": efields,
        "means": means,
        "sems": sems,
    })
    print(f"{config["label"]}: {len(means)} electric-field points")


print(study_results[0]["efields"])
efields = study_results[0]["efields"]
means = study_results[0]["means"]
sems = study_results[0]["sems"]

fit_mask = (
    (efields > 0.0)
    & np.isfinite(efields)
    & np.isfinite(means)
    & np.isfinite(sems)
    & (sems > 0.0)
)

cost = LeastSquares(
    efields[fit_mask],
    means[fit_mask],
    sems[fit_mask],
    LArQL,
)

#cost = LeastSquares(
#    efield_previous,
#    relative_s1_previous,
#    relative_s1_previous*0.01,
#    LArQL,
#)

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

with plt.rc_context({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 15,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
}):
    fig, ax = plt.subplots(
        figsize=(10, 6),
        dpi=140,
        constrained_layout=True,
    )

    # Previous study
    ax.scatter(
        efield_previous,
        relative_s1_previous,
        marker="x",
        alpha=0.9,
        color='orange',
        label="M8 previous study",
    )

    fit_text = (
    rf"$\mathbf{{Fit\ parameters:}}$"
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
    ax.text(
    0.97,
    0.97,
    fit_text,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=12,
    bbox={
        "boxstyle": "round",
        "facecolor": "white",
        "alpha": 0.85,
    },
    )

    # Current studies
    for index, study in enumerate(study_results):
        ax.errorbar(
            study["efields"],
            study["means"],
            yerr=study["sems"],
            elinewidth=1.4,
            capsize=3,
            capthick=1.2,
            fmt='*',
            color='blue',
            alpha=0.9,
            label=study["label"],
        )
    dunestyle.Preliminary(x=0.5,y=0.9)
    dunestyle.WIP(x=0.5,y=0.85)
    # Reference data
    ax.errorbar(
        E,
        PD_HD_result,
        yerr=[PD_HD_err_low, PD_HD_err_high],
        fmt="o",
        markersize=3.5,
        elinewidth=1.4,
        capsize=3,
        capthick=1.2,
        color='orange',
        label="Reference data PD-HD",
        zorder=5,
    )
    x_fit = np.linspace(0, efields.max(), 500)

    y_fit = LArQL(
    x_fit,
    m.values["B_1"],
    m.values["k_e"],
    m.values["B_2"],
    m.values["E_0"],
    )

    y_fit_gabriel = LArQL(
    x_fit,
    0.92,
    0.07,
    0.41,
    0.05,
    )

    ax.plot(x_fit, y_fit, color="blue", linewidth=2, label="This study - LArQL fit")
    ax.plot(x_fit, y_fit_gabriel, color="orange",ls='--', linewidth=2, label="Previous VD+HD - LArQL fit")


    # Reference line at zero-field normalization
    ax.axhline(
        1.0,
        color="0.45",
        linewidth=1,
        linestyle="--",
        alpha=0.6,
        zorder=1,
    )

    ax.set(
        title="Light yield versus electric field",
        xlabel=r"Electric field [kV/cm]",
        ylabel=r"Relative light yield, $S1_{\mathrm{drift}}/S1_0$",
    )

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.grid(
        which="major",
        color="0.82",
        linewidth=0.8,
    )
    ax.grid(
        which="minor",
        color="0.92",
        linewidth=0.5,
        linestyle=":",
    )

    ax.tick_params(direction="out", length=4)
    ax.margins(x=0.03, y=0.05)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        title="Data Set",
    )
    output_dir = repo_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "LY_fit.png", format='png', dpi=150)
    plt.show()
