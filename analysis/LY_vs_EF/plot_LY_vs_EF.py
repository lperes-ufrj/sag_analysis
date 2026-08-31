from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import time


# ============================================================
# Paths
# ============================================================

path_templates = Path("../../filter/templates_large_pulses/")
path_waveforms = Path("../../coincidence/selected_waveforms/")


# ============================================================
# Load templates
# ============================================================

templates_ch_2050_charge = np.trapz(
    np.loadtxt(path_templates / "template_42320_M5_1.txt")
)

templates_ch_2051_charge = np.trapz(
    np.loadtxt(path_templates / "template_42320_M5_2.txt")
)

templates_ch_2060_charge = np.trapz(
    np.loadtxt(path_templates / "template_40808_M6_1.txt")
)

templates_ch_2061_charge = np.trapz(
    np.loadtxt(path_templates / "template_40808_M6_2.txt")
)

templates_ch_2070_charge = np.trapz(
    np.loadtxt(path_templates / "template_43229_M7_1.txt")
)

templates_ch_2071_charge = np.trapz(
    np.loadtxt(path_templates / "template_43229_M7_2.txt")
)

templates_ch_2080_charge = np.trapz(
    np.loadtxt(path_templates / "template_42321_M8_1.txt")
)

templates_ch_2081_charge = np.trapz(
    np.loadtxt(path_templates / "template_42321_M8_2.txt")
)


# Template integral associated with each channel
list_templates_charge = {
    2050: templates_ch_2050_charge,
    2051: templates_ch_2051_charge,
    2060: templates_ch_2060_charge,
    2061: templates_ch_2061_charge,
    2070: templates_ch_2070_charge,
    2071: templates_ch_2071_charge,
    2080: templates_ch_2080_charge,
    2081: templates_ch_2081_charge,
}


# ============================================================
# Find waveform CSV files
# ============================================================

csv_files_0_adc = sorted(
    path_waveforms.glob(
        "channel_*_run_*coinc_2030-2031-2040-2041*min_amplitude_0_adc.csv"
    )
    
)

print(f"Found {len(csv_files_0_adc)} CSV files.")


# ============================================================
# Run -> Electric field
# ============================================================

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


# ============================================================
# ProtoDUNE-HD reference data
# ============================================================

E = np.array([
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
])

S1 = np.array([
    1.000,
    0.890,
    0.792,
    0.770,
    0.748,
    0.698,
    0.665,
    0.658,
    0.630,
    0.638,
])

S1_err_low = np.array([
    0.000,
    0.022,
    0.035,
    0.025,
    0.025,
    0.018,
    0.025,
    0.030,
    0.024,
    0.029,
])

S1_err_high = np.array([
    0.000,
    0.026,
    0.038,
    0.020,
    0.025,
    0.018,
    0.023,
    0.027,
    0.022,
    0.028,
])


# ============================================================
# Charge calculation
# ============================================================

def Calc_Charge(waveform, template_charge):
    """
    Calculate waveform charge normalized by the integral
    of the corresponding template.
    """

    waveform_charge = np.trapz(waveform[50:500])

    return waveform_charge / template_charge


# ============================================================
# Extract channel and run from filename
# ============================================================

def parse_filename(csv_file):
    """
    Expected filename structure:

    channel_2050_waveforms_run_039500_...

    Returns:
        channel : int
        run     : int

    or (None, None) if the filename cannot be parsed.
    """

    match = re.search(
        r"channel_(\d+)_waveforms_run_(\d+)",
        csv_file.name,
    )

    if not match:
        return None, None

    channel = int(match.group(1))
    run = int(match.group(2))

    return channel, run


# ============================================================
# First pass:
# Calculate S1 at E = 0 for every channel
#
# Run 39510 is the zero-field reference.
# ============================================================

s1_0_0_adc = {}

print("\n========================================")
print("Finding zero-field reference")
print("========================================")


for csv_file in csv_files_0_adc:

    channel, run = parse_filename(csv_file)

    if channel is None:
        print(f"Could not parse filename:")
        print(csv_file.name)
        continue

    if channel not in list_templates_charge:
        continue

    # Zero-field run
    if run != 39510:
        continue

    df = pd.read_csv(csv_file)

    if "mean" not in df.columns:
        print(
            f"Column 'mean' not found for "
            f"channel {channel}, run {run}"
        )
        continue

    mean_waveform = df["mean"].to_numpy()

    charge = Calc_Charge(
        mean_waveform,
        list_templates_charge[channel],
    )

    s1_0_0_adc[channel] = charge

    print(
        f"Channel {channel}: "
        f"S1(E=0) = {charge:.6f}"
    )


print("\nZero-field references:")
for channel in sorted(s1_0_0_adc):
    print(
        f"  Ch {channel}: "
        f"{s1_0_0_adc[channel]:.6f}"
    )


# ============================================================
# Second pass:
# Calculate relative S1 for every run/channel
# ============================================================

channel_points_0_adc = defaultdict(
    lambda: {
        "efield": [],
        "relative_s1": [],
        "run": [],
    }
)


print("\n========================================")
print("Calculating relative S1")
print("========================================")


for csv_file in csv_files_0_adc:

    channel, run = parse_filename(csv_file)

    if channel is None:
        continue

    # Make sure run has an E-field
    if run not in run_to_efield:
        print(
            f"Run {run} not found in "
            f"run_to_efield. Skipping."
        )
        continue

    # Make sure this channel has E=0 reference
    if channel not in s1_0_0_adc:
        print(
            f"No run 039510 reference found "
            f"for channel {channel}. Skipping."
        )
        continue

    df = pd.read_csv(csv_file)

    if "mean" not in df.columns:
        print(
            f"Column 'mean' not found for "
            f"channel {channel}, run {run}"
        )
        continue

    mean_waveform = df["mean"].to_numpy()

    charge = Calc_Charge(
        mean_waveform,
        list_templates_charge[channel],
    )

    relative_s1 = (
        charge /
        s1_0_0_adc[channel]
    )

    efield = run_to_efield[run]

    channel_points_0_adc[channel]["efield"].append(
        efield
    )

    channel_points_0_adc[channel]["relative_s1"].append(
        relative_s1
    )

    channel_points_0_adc[channel]["run"].append(
        run
    )


# ============================================================
# Print results channel by channel
# ============================================================

print("\n========================================")
print("Channel results")
print("========================================")


for channel in sorted(channel_points_0_adc):

    print(f"\nChannel {channel}")

    runs = channel_points_0_adc[channel]["run"]
    efields_channel = channel_points_0_adc[channel]["efield"]
    relative_channel = channel_points_0_adc[channel]["relative_s1"]

    for run, efield, relative_s1 in zip(
        runs,
        efields_channel,
        relative_channel,
    ):

        print(
            f"  Run {run:05d} | "
            f"E = {efield:.6f} kV/cm | "
            f"S1/S1_0 = {relative_s1:.6f}"
        )


# ============================================================
# Combine all channels by E-field
# ============================================================

points_by_efield = defaultdict(list)


for channel in sorted(channel_points_0_adc):

    efields_channel = (
        channel_points_0_adc[channel]["efield"]
    )

    relative_channel = (
        channel_points_0_adc[channel]["relative_s1"]
    )

    for efield, relative_s1 in zip(
        efields_channel,
        relative_channel,
    ):

        points_by_efield[efield].append(
            relative_s1
        )


# ============================================================
# Mean S1/S1_0 at every electric field
# ============================================================

efields = np.array(
    sorted(points_by_efield.keys()),
    dtype=float,
)

means = np.array([
    np.mean(points_by_efield[efield])
    for efield in efields
])


# Standard deviation among measurements at each field
stds = np.array([
    np.std(
        points_by_efield[efield],
        ddof=1
    )
    if len(points_by_efield[efield]) > 1
    else 0.0
    for efield in efields
])


# Number of measurements contributing to each field
n_points = np.array([
    len(points_by_efield[efield])
    for efield in efields
])


print("\n========================================")
print("Combined E-field results")
print("========================================")


for efield, mean, std, n in zip(
    efields,
    means,
    stds,
    n_points,
):

    print(
        f"E = {efield:.6f} kV/cm | "
        f"mean = {mean:.6f} | "
        f"std = {std:.6f} | "
        f"N = {n}"
    )


# ============================================================
# Previous ProtoDUNE-VD study
# ============================================================

relative_s1_previous = np.array([
    1.000,
    0.938,
    0.887,
    0.858,
    0.830,
    0.785,
    0.755,
    0.735,
    0.715,
    0.695,
    0.680,
    0.665,
    0.660,
    0.640,
    0.625,
    0.620,
    0.620,
    0.560,
])


# Previous study contains the same first E-fields
# plus one additional point at 0.556 kV/cm.
efield_previous = np.append(
    efields,
    0.556,
)


# ============================================================
# Sanity checks
# ============================================================

print("\n========================================")
print("Sanity checks")
print("========================================")

print("Channels found:")
print(sorted(channel_points_0_adc.keys()))

print("\nCurrent E-fields:")
print(efields)

print("\nNumber of current E-fields:")
print(len(efields))

print("\nPrevious study:")
print(
    f"E-field points = {len(efield_previous)}"
)

print(
    f"S1 points      = {len(relative_s1_previous)}"
)


if len(efield_previous) != len(relative_s1_previous):

    raise ValueError(
        "\nPrevious-study arrays have different sizes:\n"
        f"  E-field = {len(efield_previous)}\n"
        f"  S1      = {len(relative_s1_previous)}\n\n"
        "Check the number of unique electric-field points "
        "found in the current data."
    )


if len(efields) == 0:

    raise RuntimeError(
        "No E-field points were extracted. "
        "Check filenames and regex parsing."
    )


# ============================================================
# Plot
# ============================================================

plt.figure(
    figsize=(8, 6),
    dpi=100,
)


# Previous ProtoDUNE-VD study
plt.scatter(
    efield_previous,
    relative_s1_previous,
    marker="x",
    color="pink",
    s=60,
    label="Previous study",
)


# Current analysis
plt.scatter(
    efields,
    means,
    marker="*",
    s=100,
    label="Mean all channels and runs",
)


# ProtoDUNE-HD reference
plt.errorbar(
    E,
    S1,
    yerr=[
        S1_err_low,
        S1_err_high,
    ],
    color="black",
    fmt="o",
    capsize=3,
    label="Reference Data PD-HD",
    zorder=10,
)


# ============================================================
# Plot formatting
# ============================================================

plt.xlabel(
    "E-Field (kV/cm)"
)

plt.ylabel(
    r"$S1_{\mathrm{drift}} / S1_{0}$"
)

plt.title(
    "LY vs E-Field"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()


# ============================================================
# Save
# ============================================================

output_file = (
    f"LY_vs_EF_{int(time.time())}.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

print(
    f"\nPlot saved as: {output_file}"
)


plt.show()