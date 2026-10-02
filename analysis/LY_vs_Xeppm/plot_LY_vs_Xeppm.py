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

script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parents[1]
path_templates = repo_dir / "filter" / "templates_large_pulses"
path_waveforms = repo_dir / "coincidence" / "selected_waveforms" / "20261001_140103"


# ============================================================
# Load templates
# ============================================================

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


# Template integral associated with each channel
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


# ============================================================
# Find waveform CSV files
# ============================================================

csv_files = sorted(
    path_waveforms.glob(
        "channel_*_run_*.csv"
    )
    
)

CHANNELS_TO_PLOT = {2050,2051,2060,2061}

print(f"Found {len(csv_files)} CSV files.")


# ============================================================
# Run -> Xe concentration mapping
# ============================================================

run_to_Xeppm = {
    43440 : 0.01,
    43552 : 1.0,
    43717 : 2.0,
    43790 : 3.0,
    43903 : 5.0,
    44010 : 7.0,
    44108 : 10.0
}

REFERENCE_RUN = 43440
REFERENCE_XE_PPM = run_to_Xeppm[REFERENCE_RUN]


# ============================================================
# Charge calculation
# ============================================================

def Calc_Charge(waveform, template_charge):
    """
    Calculate waveform charge normalized by the integral
    of the corresponding template.
    """

    waveform = np.asarray(waveform, dtype=float)
    if waveform.size < 500 or not np.all(np.isfinite(waveform[50:500])):
        raise ValueError("Expected at least 600 samples with a finite integration window [50:600].")
    if not np.isfinite(template_charge) or template_charge == 0:
        raise ValueError("Template charge must be finite and nonzero.")
    waveform_charge = np.trapezoid(waveform[50:500])

    return waveform_charge / template_charge


# ============================================================
# Extract channel and run from filename
# ============================================================

def parse_filename(csv_file):
    """
    Expected filename structure:

    channel_2080_coincidence_scan_run_043440_...

    Returns:
        channel : int
        run     : int

    or (None, None) if the filename cannot be parsed.
    """

    match = re.search(
        r"channel_(\d+)_coincidence_scan_run_(\d+)",
        csv_file.name,
    )

    if not match:
        return None, None

    channel = int(match.group(1))
    run = int(match.group(2))



    return channel, run


# ============================================================
# First pass:
# Calculate the 0.01 ppm reference S1 separately for every channel.
#
# Run 43440 is the reference sample.
# ============================================================

reference_s1 = {}

print("\n========================================")
print(f"Finding {REFERENCE_XE_PPM:g} ppm reference (run {REFERENCE_RUN:06d})")
print("========================================")


for csv_file in csv_files:

    channel, run = parse_filename(csv_file)

    if channel not in CHANNELS_TO_PLOT:
        continue

    if channel is None:
        print(f"Could not parse filename:")
        print(csv_file.name)
        continue

    if channel not in list_templates_charge:
        continue

    # Xe 0.01 ppm run
    if run != REFERENCE_RUN:
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

    if not np.isfinite(charge) or charge == 0:
        raise ValueError(f"Invalid reference charge for channel {channel}: {charge}")
    if channel in reference_s1:
        raise ValueError(f"Multiple reference CSVs for channel {channel}, run {REFERENCE_RUN}.")
    reference_s1[channel] = charge

    print(
        f"Channel {channel}: "
        f"S1({REFERENCE_XE_PPM:g} ppm) = {charge:.6f}"
    )


print("\n0.01 ppm references:")
for channel in sorted(reference_s1):
    print(
        f"  Ch {channel}: "
        f"{reference_s1[channel]:.6f}"
    )


# ============================================================
# Check every requested channel has its own reference.
missing_references = CHANNELS_TO_PLOT - reference_s1.keys()
if missing_references:
    raise RuntimeError(
        f"Missing {REFERENCE_XE_PPM:g} ppm reference (run {REFERENCE_RUN:06d}) "
        f"for channels {sorted(missing_references)} in {path_waveforms}"
    )

# Second pass:
# Calculate relative S1 for every run/channel
# ============================================================

channel_points = defaultdict(
    lambda: {
        "xeppm": [],
        "relative_s1": [],
        "run": [],
    }
)


print("\n========================================")
print("Calculating relative S1")
print("========================================")

for csv_file in csv_files:

    channel, run = parse_filename(csv_file)

    if channel not in CHANNELS_TO_PLOT:
        continue

    if channel is None:
        continue
    
    print(csv_file)

    # Make sure the run has a Xe concentration
    if run not in run_to_Xeppm:
        print(
            f"Run {run} not found in "
            f"run_to_Xeppm. Skipping."
        )
        continue

    # Make sure this channel has a 0.01 ppm reference
    if channel not in reference_s1:
        print(
            f"No run {REFERENCE_RUN:06d} reference found "
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
        reference_s1[channel]
    )

    if run in channel_points[channel]["run"]:
        raise ValueError(f"Multiple CSVs for channel {channel}, run {run}.")
    xeppm = run_to_Xeppm[run]

    channel_points[channel]["xeppm"].append(
        xeppm
    )

    channel_points[channel]["relative_s1"].append(
        relative_s1
    )

    channel_points[channel]["run"].append(
        run
    )


# ============================================================
# Print results channel by channel
# ============================================================

print("\n========================================")
print("Channel results")
print("========================================")


for channel in sorted(channel_points):

    print(f"\nChannel {channel}")

    runs = channel_points[channel]["run"]
    xeppm_channel = channel_points[channel]["xeppm"]
    relative_channel = channel_points[channel]["relative_s1"]

    for run, xeppm, relative_s1 in zip(
        runs,
        xeppm_channel,
        relative_channel,
    ):

        print(
            f"  Run {run:05d} | "
            f"Xe = {xeppm:.6f} ppm | "
            f"S1/S1_0 = {relative_s1:.6f}"
        )


# ============================================================
# Combine normalized channels by Xe concentration
# ============================================================

points_by_xeppm = defaultdict(list)


for channel in sorted(channel_points):

    xeppm_channel = (
        channel_points[channel]["xeppm"]
    )

    relative_channel = (
        channel_points[channel]["relative_s1"]
    )

    for xeppm, relative_s1 in zip(
        xeppm_channel,
        relative_channel,
    ):

        points_by_xeppm[xeppm].append(
            relative_s1
        )

print(f"Found {len(csv_files)} CSV files.")
# ============================================================
# Mean S1/S1(0.01 ppm) at every Xe concentration
# ============================================================

xeppm_values = np.array(
    sorted(points_by_xeppm.keys()),
    dtype=float,
)

means = np.array([
    np.mean(points_by_xeppm[xeppm])
    for xeppm in xeppm_values
])


# Standard deviation between channels, not statistical uncertainty
stds = np.array([
    np.std(
        points_by_xeppm[xeppm],
        ddof=1
    )
    if len(points_by_xeppm[xeppm]) > 1
    else 0.0
    for xeppm in xeppm_values
])


# Number of channels contributing to each concentration
n_points = np.array([
    len(points_by_xeppm[xeppm])
    for xeppm in xeppm_values
])


print("\n========================================")
print("Combined Xe concentration results")
print("========================================")


for xeppm, mean, std, n in zip(
    xeppm_values,
    means,
    stds,
    n_points,
):

    print(
        f"Xe = {xeppm:.6f} ppm | "
        f"mean = {mean:.6f} | "
        f"std = {std:.6f} | "
        f"N = {n}"
    )




# ============================================================
# Sanity checks
# ============================================================

print("\n========================================")
print("Sanity checks")
print("========================================")

print("Channels found:")
print(sorted(channel_points.keys()))

print("\nCurrent Xe Concentrations:")
print(xeppm_values)

print("\nNumber of current Xe concentrations:")
print(len(xeppm_values))



if len(xeppm_values) == 0:

    raise RuntimeError(
        "No Xe points were extracted. "
        "Check filenames and regex parsing."
    )


# ============================================================
# Plot
# ============================================================

if np.any(n_points != len(CHANNELS_TO_PLOT)):
    raise RuntimeError("Each Xe concentration must contain all requested channels for a consistent mean.")

plt.figure(figsize=(8, 6),dpi=100)

# Normalize each channel before averaging, so channels have equal weight.
for channel in sorted(channel_points):
    points = channel_points[channel]
    order = np.argsort(points["xeppm"])
    plt.plot(
        np.asarray(points["xeppm"])[order],
        np.asarray(points["relative_s1"])[order],
        "o--",
        alpha=0.6,
        label=f"Channel {channel}",
    )

plt.errorbar(
    xeppm_values,
    means,
    yerr=stds,
    fmt="*-",
    markersize=10,
    color="black",
    capsize=4,
    label="Channel mean ± channel standard deviation",
)
plt.axhline(1.0, color="gray", linestyle=":", linewidth=1)
plt.xlabel("Xe Concentration (ppm)")
plt.ylabel(r"Relative light yield $S1(c_{\mathrm{Xe}}) / S1(0.01\,\mathrm{ppm})$")
plt.title(f"LY vs Xe concentration (sample {path_waveforms.name})")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

output_file = (script_dir / f"LY_vs_Xe_{int(time.time())}.png")
plt.savefig(output_file,dpi=300,bbox_inches="tight",)
print(f"\nPlot saved as: {output_file}")
plt.show()