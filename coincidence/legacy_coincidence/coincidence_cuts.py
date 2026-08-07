import ROOT
import numpy as np
import argparse
import sys
from array import array
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

f_list = [
    str(REPO_DIR / "np02vd_raw_run039510_0000_df-s04-d0_dw_0_20250919T123428_0kV_gallery.root"),
    str(REPO_DIR / "np02vd_raw_run039510_0001_df-s04-d0_dw_0_20250919T123526_0kV_gallery.root"),
    str(REPO_DIR / "np02vd_raw_run039510_0002_df-s04-d0_dw_0_20250919T123624_0kV_gallery.root"),
    str(REPO_DIR / "np02vd_raw_run039510_0003_df-s04-d0_dw_0_20250919T123722_0kV_gallery.root"),
    str(REPO_DIR / "np02vd_raw_run039510_0004_df-s04-d0_dw_0_20250919T123821_0kV_gallery.root"),
    str(REPO_DIR / "np02vd_raw_run039510_0005_df-s04-d0_dw_0_20250919T123919_0kV_gallery.root"),
]


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
channels_dict = {
    "CH_M3_1": 2030, 
    "CH_M3_2": 2031, 
    "CH_M4_1": 2040, 
    "CH_M4_2": 2041, 
    "CH_M5_1": 2050, 
    "CH_M5_2": 2051, 
    "CH_M6_1": 2060, 
    "CH_M6_2": 2061,
    "CH_M7_1": 2070,
    "CH_M7_2": 2071,
    "CH_M8_1": 2080,
    "CH_M8_2": 2081
}

DEFAULT_COINCIDENCE_WINDOW_TICKS = 20
DEFAULT_COINCIDENCE_THRESHOLD_ADC = 8000
DEFAULT_SAVE_THRESHOLD_ADC = 3000

all_channels = np.array(list(channels_dict.values()), dtype=np.int64)
channels_for_coincidence_left = all_channels[:4]
channels_for_coincidence_right = all_channels[4:8]
channels_to_save_wf = all_channels[-4:]

def baseline(waveform):
    """Calculate the baseline of a waveform."""
    return np.mean(waveform[0:300])

def signal_amplitude(waveform):
    waveform = np.asarray(waveform)
    signal_region = waveform[30:130]
    return np.max(signal_region) - np.min(signal_region)

def signal_peak_tick(waveform):
    waveform = np.asarray(waveform)
    signal_region = waveform[30:130]
    return 30 + int(np.argmax(signal_region))

def noise_rms(waveform):
    """Calculate the RMS of the noise in a waveform."""
    return np.std(waveform[0:300])

def preSignalAmplitude(waveform):
    """Calculate the pre-signal amplitude of a waveform."""
    return np.max(waveform[0:300]) - np.min(waveform[0:300])

def postSignalAmplitude(waveform):
    """Calculate the post-signal amplitude of a waveform."""
    return np.max(waveform[350:1023]) - np.min(waveform[350:1023])

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Find coincident waveform pairs using a peak-tick window and save "
            "nearby waveforms from selected channels."
        )
    )
    parser.add_argument(
        "--coincidence-window-ticks",
        "--window-ticks",
        type=int,
        default=DEFAULT_COINCIDENCE_WINDOW_TICKS,
        help="Allowed difference between left and right peak ticks.",
    )
    parser.add_argument(
        "--coincidence-threshold-adc",
        "--threshold-adc-coincidence",
        type=float,
        default=DEFAULT_COINCIDENCE_THRESHOLD_ADC,
        help="Minimum waveform amplitude used to form coincidence pairs.",
    )
    parser.add_argument(
        "--save-threshold-adc",
        "--threshold-adc-save",
        type=float,
        default=DEFAULT_SAVE_THRESHOLD_ADC,
        help="Minimum waveform amplitude required before saving a waveform.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output ROOT file name.",
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        default=f_list,
        help="Input ROOT files.",
    )
    return parser

def parse_args():
    parser = build_parser()
    args = parser.parse_args()

    if args.coincidence_window_ticks < 0:
        parser.error("--coincidence-window-ticks must be >= 0")
    if args.coincidence_threshold_adc < 0:
        parser.error("--coincidence-threshold-adc must be >= 0")
    if args.save_threshold_adc < 0:
        parser.error("--save-threshold-adc must be >= 0")

    if args.output is None:
        args.output = (
            "coincident_waveforms_fast"
            f"_window_{args.coincidence_window_ticks}_ticks"
            f"_coinc_adc_{args.coincidence_threshold_adc:g}"
            f"_save_adc_{args.save_threshold_adc:g}"
            ".root"
        )

    return args

args = parse_args()

chain = ROOT.TChain("WaveformTree")

for f_in in args.input:
    chain.Add(f_in)

n_entries = chain.GetEntries()
print(f"Input tree entries: {n_entries}")

if n_entries == 0:
    print(
        "No entries found in WaveformTree. Check that the input ROOT files exist "
        "and contain the expected tree.",
        file=sys.stderr,
    )
    sys.exit(1)

chain.SetBranchStatus("*", 1)
df = ROOT.RDataFrame(chain)

valid_channels = np.unique(np.concatenate([
    channels_for_coincidence_left,
    channels_for_coincidence_right,
    channels_to_save_wf,
]))
channel_filter = " || ".join(f"channel == {int(ch)}" for ch in valid_channels)

arr = (
    df.Filter(channel_filter)
      .AsNumpy(["event", "channel", "waveform_index", "adc", "timestamp"])
)

events = arr["event"].astype(np.int64, copy=False)
timestamps = arr["timestamp"].astype(np.int64, copy=False)
wf = arr["adc"]
wf_index = arr["waveform_index"].astype(np.int64, copy=False)
channel_arr = arr["channel"].astype(np.int64, copy=False)

threshold_adc_coincidence = args.coincidence_threshold_adc
threshold_adc_save = args.save_threshold_adc
window_ticks = args.coincidence_window_ticks
output_name = args.output

print(f"Total waveforms collected: {len(events)}")
print(f"Coincidence-left channels: {channels_for_coincidence_left.tolist()}")
print(f"Coincidence-right channels: {channels_for_coincidence_right.tolist()}")
print(f"Saved waveform channels: {channels_to_save_wf.tolist()}")
print(f"Coincidence window: {window_ticks} ticks")
print(f"Coincidence amplitude threshold: {threshold_adc_coincidence:g} ADC")
print(f"Saved waveform amplitude threshold: {threshold_adc_save:g} ADC")

# -------------------------------------------------------
# Precompute waveform quantities once
# -------------------------------------------------------
n_waveforms = len(events)

amplitude = np.empty(n_waveforms, dtype=np.float64)
peak_tick = np.empty(n_waveforms, dtype=np.int32)

for i in range(n_waveforms):
    wf_i = np.asarray(wf[i])
    amplitude[i] = signal_amplitude(wf_i)
    peak_tick[i] = signal_peak_tick(wf_i)

coincidence_good = amplitude > threshold_adc_coincidence
save_good = amplitude > threshold_adc_save

def build_lookup(mask):
    lookup = defaultdict(list)
    for i in np.where(mask)[0]:
        lookup[(int(events[i]), int(channel_arr[i]))].append(i)
    return lookup

def sort_lookup_by_peak(lookup):
    sorted_lookup = {}
    for key, indices in lookup.items():
        indices = np.array(indices, dtype=np.int64)
        order = np.argsort(peak_tick[indices])
        sorted_indices = indices[order]
        sorted_lookup[key] = (sorted_indices, peak_tick[sorted_indices])
    return sorted_lookup

# Coincidence channels use the high ADC threshold.
by_event_channel_coincidence = sort_lookup_by_peak(build_lookup(coincidence_good))

# Saved waveforms use the lower ADC threshold.
by_event_channel_save = sort_lookup_by_peak(build_lookup(save_good))

# -------------------------------------------------------
# Create output ROOT file/tree
# -------------------------------------------------------
out_file = ROOT.TFile(output_name, "RECREATE")
out_tree = ROOT.TTree("CoincidentWaveforms", "Waveforms passing coincidence selection")

pair_id = array("q", [0])
event_out = array("q", [0])

coincidence_left_channel = array("i", [0])
coincidence_right_channel = array("i", [0])
coincidence_left_waveform_index = array("q", [0])
coincidence_right_waveform_index = array("q", [0])
coincidence_left_timestamp = array("q", [0])
coincidence_right_timestamp = array("q", [0])
dt_timestamp_ticks = array("q", [0])
dt_peak_ticks = array("i", [0])

saved_channel = array("i", [0])
saved_waveform_index = array("q", [0])
saved_timestamp = array("q", [0])
saved_amplitude = array("d", [0.0])
saved_peak_tick = array("i", [0])

adc_vec = ROOT.std.vector("double")()

out_tree.Branch("pair_id", pair_id, "pair_id/L")
out_tree.Branch("event", event_out, "event/L")

out_tree.Branch("coincidence_left_channel", coincidence_left_channel, "coincidence_left_channel/I")
out_tree.Branch("coincidence_right_channel", coincidence_right_channel, "coincidence_right_channel/I")
out_tree.Branch("coincidence_left_waveform_index", coincidence_left_waveform_index, "coincidence_left_waveform_index/L")
out_tree.Branch("coincidence_right_waveform_index", coincidence_right_waveform_index, "coincidence_right_waveform_index/L")
out_tree.Branch("coincidence_left_timestamp", coincidence_left_timestamp, "coincidence_left_timestamp/L")
out_tree.Branch("coincidence_right_timestamp", coincidence_right_timestamp, "coincidence_right_timestamp/L")
out_tree.Branch("dt_timestamp_ticks", dt_timestamp_ticks, "dt_timestamp_ticks/L")
out_tree.Branch("dt_peak_ticks", dt_peak_ticks, "dt_peak_ticks/I")

out_tree.Branch("saved_channel", saved_channel, "saved_channel/I")
out_tree.Branch("saved_waveform_index", saved_waveform_index, "saved_waveform_index/L")
out_tree.Branch("saved_timestamp", saved_timestamp, "saved_timestamp/L")
out_tree.Branch("saved_amplitude", saved_amplitude, "saved_amplitude/D")
out_tree.Branch("saved_peak_tick", saved_peak_tick, "saved_peak_tick/I")
out_tree.Branch("adc", adc_vec)

def fill_saved_waveform(i_save):
    saved_channel[0] = int(channel_arr[i_save])
    saved_waveform_index[0] = int(wf_index[i_save])
    saved_timestamp[0] = int(timestamps[i_save])
    saved_amplitude[0] = float(amplitude[i_save])
    saved_peak_tick[0] = int(peak_tick[i_save])

    adc_vec.clear()
    wf_save = np.asarray(wf[i_save])
    for adc_value in wf_save:
        adc_vec.push_back(float(adc_value))

    out_tree.Fill()

# -------------------------------------------------------
# Fast matching and writing
# -------------------------------------------------------
n_pairs = 0
n_saved_waveforms = 0

for evt in np.unique(events):

    for chLeft in channels_for_coincidence_left:
        left_sorted, _ = by_event_channel_coincidence.get(
            (int(evt), int(chLeft)),
            (np.array([], dtype=np.int64), np.array([], dtype=np.int32))
        )

        if len(left_sorted) == 0:
            continue

        for chRight in channels_for_coincidence_right:
            right_sorted, right_ticks_sorted = by_event_channel_coincidence.get(
                (int(evt), int(chRight)),
                (np.array([], dtype=np.int64), np.array([], dtype=np.int32))
            )

            if len(right_sorted) == 0:
                continue

            for i_left in left_sorted:
                t_left = peak_tick[i_left]

                lo = np.searchsorted(
                    right_ticks_sorted,
                    t_left - window_ticks,
                    side="left"
                )
                hi = np.searchsorted(
                    right_ticks_sorted,
                    t_left + window_ticks,
                    side="right"
                )

                for i_right in right_sorted[lo:hi]:
                    t_right = peak_tick[i_right]
                    save_window_min = min(t_left, t_right) - window_ticks
                    save_window_max = max(t_left, t_right) + window_ticks

                    pair_id[0] = n_pairs
                    event_out[0] = int(evt)

                    coincidence_left_channel[0] = int(chLeft)
                    coincidence_right_channel[0] = int(chRight)

                    coincidence_left_waveform_index[0] = int(wf_index[i_left])
                    coincidence_right_waveform_index[0] = int(wf_index[i_right])

                    coincidence_left_timestamp[0] = int(timestamps[i_left])
                    coincidence_right_timestamp[0] = int(timestamps[i_right])

                    dt_timestamp_ticks[0] = int(timestamps[i_right] - timestamps[i_left])
                    dt_peak_ticks[0] = int(t_right - t_left)

                    for chSave in channels_to_save_wf:
                        save_sorted, save_ticks_sorted = by_event_channel_save.get(
                            (int(evt), int(chSave)),
                            (np.array([], dtype=np.int64), np.array([], dtype=np.int32))
                        )

                        if len(save_sorted) == 0:
                            continue

                        lo_save = np.searchsorted(
                            save_ticks_sorted,
                            save_window_min,
                            side="left"
                        )
                        hi_save = np.searchsorted(
                            save_ticks_sorted,
                            save_window_max,
                            side="right"
                        )

                        for i_save in save_sorted[lo_save:hi_save]:
                            fill_saved_waveform(i_save)
                            n_saved_waveforms += 1

                    n_pairs += 1

out_file.cd()
out_tree.Write()
out_file.Close()

print(f"Saved {n_saved_waveforms} waveform entries")
print(f"Saved {n_pairs} coincident pairs")
print(f"Output file: {output_name}")
